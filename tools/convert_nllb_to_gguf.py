#!/usr/bin/env python3
"""
Convert facebook/nllb-200-distilled-600M (HF M2M100 architecture) to GGUF.

Reads either model.safetensors or pytorch_model.bin (sharded or not).
safetensors needs no torch; the .bin path imports torch lazily, only if a .bin
is what you actually have.

    pip install numpy safetensors gguf sentencepiece
    python tools/convert_nllb_to_gguf.py models/nllb-600M \
        --outfile models/nllb-600M-f32.gguf --outtype f32

Design notes that matter for a from-scratch C++ port:

  * Linear weights are written WITHOUT transposing. torch nn.Linear stores
    (out_features, in_features); gguf reverses the shape on write, so ggml sees
    ne = [in_features, out_features], which is exactly what ggml_mul_mat wants.

  * Position embeddings in M2M100 are SINUSOIDAL. Depending on the transformers
    version that produced the checkpoint they are either absent (non-persistent
    buffer) or present as `model.encoder.embed_positions.weights`. We recompute
    the table the way HF does, use the checkpoint's copy when there is one, and
    cross-check the two. Either way it is BAKED IN as `position_embd.weight` so
    the C++ side does a row lookup and no trig at all.

  * Position index for token j of an unpadded sequence is j + 2, not j.
    (HF: cumsum(mask)*mask + padding_idx, with padding_idx=1 and offset=2.)

  * Token embeddings are scaled by sqrt(d_model) BEFORE adding positions.
    Positions are NOT scaled.

  * Attention scaling (head_dim ** -0.5) is applied to Q in HF, not to the
    logits. Mathematically equivalent, but match it if you want intermediate
    tensors to line up when you bisect a mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")


ARCH = "nllb"


# --------------------------------------------------------------------------
# checkpoint loading: safetensors or pytorch .bin, sharded or single
# --------------------------------------------------------------------------

class Checkpoint:
    """Uniform read-only view over an HF checkpoint directory.

    Prefers safetensors (mmap'd, no torch). Falls back to pytorch_model.bin,
    which is a pickle and therefore needs torch to unpickle.
    """

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.backend = ""
        self.map: dict[str, Path] = {}      # safetensors: name -> shard
        self.sd: dict[str, object] = {}     # torch: name -> tensor
        self._handles: dict[Path, object] = {}

        st_index = model_dir / "model.safetensors.index.json"
        st_single = model_dir / "model.safetensors"
        bin_index = model_dir / "pytorch_model.bin.index.json"
        bin_single = model_dir / "pytorch_model.bin"

        st_shards = sorted(model_dir.glob("*.safetensors"))
        bin_shards = sorted(p for p in model_dir.glob("*.bin")
                            if "training_args" not in p.name)

        if st_index.exists() or st_single.exists() or st_shards:
            self._init_safetensors(st_index, st_single, st_shards)
        elif bin_index.exists() or bin_single.exists() or bin_shards:
            self._init_torch(bin_index, bin_single, bin_shards)
        else:
            raise FileNotFoundError(
                f"no model weights in {model_dir}. Expected model.safetensors "
                f"or pytorch_model.bin. Found: "
                f"{sorted(p.name for p in model_dir.iterdir())[:12]}"
            )

    # -- safetensors --------------------------------------------------------

    def _init_safetensors(self, index: Path, single: Path, shards: list[Path]):
        try:
            from safetensors import safe_open
        except ImportError:
            sys.exit("pip install safetensors")
        self._safe_open = safe_open
        self.backend = "safetensors"

        if index.exists():
            with open(index) as f:
                for name, shard in json.load(f)["weight_map"].items():
                    self.map[name] = self.dir / shard
            return
        for path in ([single] if single.exists() else shards):
            with safe_open(path, framework="np") as f:
                for name in f.keys():
                    self.map[name] = path

    # -- torch .bin ---------------------------------------------------------

    def _init_torch(self, index: Path, single: Path, shards: list[Path]):
        try:
            import torch
        except ImportError:
            sys.exit(
                "This checkpoint is pytorch_model.bin (a pickle), which needs "
                "torch to read.\n"
                "  pip install torch\n"
                "Or download the safetensors revision instead, if the repo has one."
            )
        self.backend = "pytorch_model.bin"

        if index.exists():
            with open(index) as f:
                files = sorted({self.dir / s
                                for s in json.load(f)["weight_map"].values()})
        elif single.exists():
            files = [single]
        else:
            files = shards

        for path in files:
            print(f"  loading {path.name} ...", flush=True)
            try:
                part = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:                      # torch < 1.13
                part = torch.load(path, map_location="cpu")
            if not isinstance(part, dict):
                raise TypeError(f"{path.name} is not a state dict")
            # some exports wrap the state dict
            if "state_dict" in part and isinstance(part["state_dict"], dict):
                part = part["state_dict"]
            self.sd.update(part)

    # -- access -------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.map or name in self.sd

    def keys(self):
        return self.map.keys() if self.backend == "safetensors" else self.sd.keys()

    def get(self, name: str) -> np.ndarray:
        if self.backend == "safetensors":
            if name not in self.map:
                raise KeyError(name)
            path = self.map[name]
            if path not in self._handles:
                self._handles[path] = self._safe_open(path, framework="np")
            return np.asarray(self._handles[path].get_tensor(name))

        if name not in self.sd:
            raise KeyError(name)
        t = self.sd[name]
        # .float() first: bfloat16 has no numpy equivalent
        return t.detach().to("cpu").float().numpy()

    def first(self, *names: str) -> np.ndarray:
        """Return the first name that exists. HF drops duplicates of tied
        weights in safetensors, so the shared embedding may live under any of
        several aliases; in .bin checkpoints all the aliases are present and
        point at the same storage."""
        for n in names:
            if n in self:
                return self.get(n)
        raise KeyError(f"none of {names} found. keys sample: "
                       f"{sorted(self.keys())[:8]}")


# --------------------------------------------------------------------------
# sinusoidal position table, reproduced exactly from HF M2M100
# --------------------------------------------------------------------------

def build_sinusoidal_table(num_positions: int, dim: int, padding_idx: int) -> np.ndarray:
    """Bit-for-bit port of M2M100SinusoidalPositionalEmbedding.get_embedding.

    Note the (half_dim - 1) denominator and the sin|cos CONCATENATION
    (first half sin, second half cos) - not the interleaved layout used by
    the original 'Attention is All You Need' code and by most blog posts.
    """
    half = dim // 2
    step = math.log(10000.0) / (half - 1)
    inv = np.exp(np.arange(half, dtype=np.float64) * -step)
    pos = np.arange(num_positions, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([np.sin(pos), np.cos(pos)], axis=1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros((num_positions, 1))], axis=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0.0
    return emb.astype(np.float32)


def resolve_position_table(ck: Checkpoint, max_pos: int, d_model: int,
                           pad_id: int) -> np.ndarray:
    """Recompute the table, and cross-check against the checkpoint's own copy
    if this transformers version happened to serialise one."""
    computed = build_sinusoidal_table(max_pos + 2, d_model, padding_idx=pad_id)

    stored = None
    for key in ("model.encoder.embed_positions.weights",
                "model.encoder.embed_positions.weight"):
        if key in ck:
            stored = ck.get(key)
            break
    if stored is None:
        return computed

    n = min(len(stored), len(computed))
    d = float(np.abs(stored[:n] - computed[:n]).max())
    if d > 1e-4:
        print(f"  ! position table differs from checkpoint copy (max|d|={d:.2e})"
              f" - using the checkpoint's", file=sys.stderr)
        out = computed.copy()
        out[:n] = stored[:n]
        return out
    print(f"  position table matches checkpoint copy (max|d|={d:.2e})")
    return computed


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

def load_vocab(model_dir: Path):
    """Return (tokens, scores, toktypes, meta) or None.

    Prefers tokenizer.json, whose vocab dict already has NLLB's fairseq +1 id
    offset and the 200+ FLORES language codes baked in - so we never have to
    guess at the offset arithmetic.
    """
    tok_json = model_dir / "tokenizer.json"
    spm_file = model_dir / "sentencepiece.bpe.model"

    if tok_json.exists():
        with open(tok_json, encoding="utf-8") as f:
            tj = json.load(f)
        vocab = tj["model"]["vocab"]
        added = {t["id"]: t for t in tj.get("added_tokens", [])}
        size = max(max(vocab.values()), max(added) if added else -1) + 1

        tokens = [f"[UNUSED_{i}]" for i in range(size)]
        scores = [0.0] * size
        toktypes = [gguf.TokenType.UNUSED] * size

        spm_scores: dict[str, float] = {}
        if spm_file.exists():
            try:
                import sentencepiece as spmlib
                sp = spmlib.SentencePieceProcessor(model_file=str(spm_file))
                for i in range(sp.GetPieceSize()):
                    spm_scores[sp.IdToPiece(i)] = sp.GetScore(i)
            except ImportError:
                print("  ! sentencepiece not installed - vocab scores will be 0",
                      file=sys.stderr)

        for piece, idx in vocab.items():
            tokens[idx] = piece
            scores[idx] = spm_scores.get(piece, 0.0)
            toktypes[idx] = gguf.TokenType.NORMAL
        for idx, t in added.items():
            tokens[idx] = t["content"]
            toktypes[idx] = (gguf.TokenType.CONTROL if t.get("special")
                             else gguf.TokenType.USER_DEFINED)
        return tokens, scores, toktypes, {"source": "tokenizer.json"}

    if not spm_file.exists():
        return None

    import sentencepiece as spmlib
    sp = spmlib.SentencePieceProcessor(model_file=str(spm_file))

    offset = 1                      # HF NllbTokenizer.fairseq_offset
    n_spm = sp.GetPieceSize()
    tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    scores = [0.0] * 4
    toktypes = [gguf.TokenType.CONTROL] * 4

    for i in range(n_spm):
        idx = i + offset
        if idx < 4:
            continue
        tokens.append(sp.IdToPiece(i))
        scores.append(sp.GetScore(i))
        if sp.IsUnknown(i) or sp.IsControl(i):
            tt = gguf.TokenType.CONTROL
        elif sp.IsUnused(i):
            tt = gguf.TokenType.UNUSED
        elif sp.IsByte(i):
            tt = gguf.TokenType.BYTE
        else:
            tt = gguf.TokenType.NORMAL
        toktypes.append(tt)

    stm_path = model_dir / "special_tokens_map.json"
    langs: list[str] = []
    if stm_path.exists():
        with open(stm_path, encoding="utf-8") as f:
            stm = json.load(f)
        langs = [t if isinstance(t, str) else t["content"]
                 for t in stm.get("additional_special_tokens", [])]
    for code in langs:
        tokens.append(code)
        scores.append(0.0)
        toktypes.append(gguf.TokenType.USER_DEFINED)
    tokens.append("<mask>")
    scores.append(0.0)
    toktypes.append(gguf.TokenType.USER_DEFINED)

    return tokens, scores, toktypes, {"source": "sentencepiece.bpe.model"}


# --------------------------------------------------------------------------
# tensor plan
# --------------------------------------------------------------------------

ATTN = [
    ("self_attn.q_proj.weight",       "attn_q.weight"),
    ("self_attn.q_proj.bias",         "attn_q.bias"),
    ("self_attn.k_proj.weight",       "attn_k.weight"),
    ("self_attn.k_proj.bias",         "attn_k.bias"),
    ("self_attn.v_proj.weight",       "attn_v.weight"),
    ("self_attn.v_proj.bias",         "attn_v.bias"),
    ("self_attn.out_proj.weight",     "attn_out.weight"),
    ("self_attn.out_proj.bias",       "attn_out.bias"),
    ("self_attn_layer_norm.weight",   "attn_norm.weight"),
    ("self_attn_layer_norm.bias",     "attn_norm.bias"),
]

CROSS = [
    ("encoder_attn.q_proj.weight",     "cross_attn_q.weight"),
    ("encoder_attn.q_proj.bias",       "cross_attn_q.bias"),
    ("encoder_attn.k_proj.weight",     "cross_attn_k.weight"),
    ("encoder_attn.k_proj.bias",       "cross_attn_k.bias"),
    ("encoder_attn.v_proj.weight",     "cross_attn_v.weight"),
    ("encoder_attn.v_proj.bias",       "cross_attn_v.bias"),
    ("encoder_attn.out_proj.weight",   "cross_attn_out.weight"),
    ("encoder_attn.out_proj.bias",     "cross_attn_out.bias"),
    ("encoder_attn_layer_norm.weight", "cross_attn_norm.weight"),
    ("encoder_attn_layer_norm.bias",   "cross_attn_norm.bias"),
]

FFN = [
    ("fc1.weight",              "ffn_up.weight"),
    ("fc1.bias",                "ffn_up.bias"),
    ("fc2.weight",              "ffn_down.weight"),
    ("fc2.bias",                "ffn_down.bias"),
    ("final_layer_norm.weight", "ffn_norm.weight"),
    ("final_layer_norm.bias",   "ffn_norm.bias"),
]


def should_keep_f32(name: str) -> bool:
    """Norms, biases and the position table stay F32 even in f16 mode.
    They are tiny and they are exactly where precision loss compounds."""
    return (name.endswith(".bias")
            or "_norm." in name
            or name.startswith("position_embd"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path,
                    help="HF snapshot dir (config.json + weights + tokenizer files)")
    ap.add_argument("--outfile", type=Path, default=None)
    ap.add_argument("--outtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--encoder-only", action="store_true",
                    help="skip decoder tensors - smaller file for milestone 1")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    md = args.model_dir
    with open(md / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    d_model   = cfg["d_model"]
    n_vocab   = cfg["vocab_size"]
    max_pos   = cfg["max_position_embeddings"]
    pad_id    = cfg.get("pad_token_id", 1)
    bos_id    = cfg.get("bos_token_id", 0)
    eos_id    = cfg.get("eos_token_id", 2)
    act       = cfg.get("activation_function", "relu")
    scale_emb = bool(cfg.get("scale_embedding", True))
    if act != "relu":
        print(f"  ! activation is {act!r}, not relu - check your C++ kernel",
              file=sys.stderr)

    out = args.outfile or Path(f"nllb-{d_model}d-{args.outtype}.gguf")
    out.parent.mkdir(parents=True, exist_ok=True)

    ck = Checkpoint(md)
    print(f"  backend: {ck.backend}")
    w = GGUFWriter(str(out), ARCH)

    # ---- metadata ---------------------------------------------------------
    w.add_name(args.name or md.name)
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if args.outtype == "f32"
                    else gguf.LlamaFileType.MOSTLY_F16)
    w.add_context_length(max_pos)
    w.add_embedding_length(d_model)
    w.add_vocab_size(n_vocab)

    w.add_uint32(f"{ARCH}.encoder.block_count",          cfg["encoder_layers"])
    w.add_uint32(f"{ARCH}.encoder.attention.head_count",  cfg["encoder_attention_heads"])
    w.add_uint32(f"{ARCH}.encoder.feed_forward_length",   cfg["encoder_ffn_dim"])
    if not args.encoder_only:
        w.add_uint32(f"{ARCH}.decoder.block_count",         cfg["decoder_layers"])
        w.add_uint32(f"{ARCH}.decoder.attention.head_count", cfg["decoder_attention_heads"])
        w.add_uint32(f"{ARCH}.decoder.feed_forward_length",  cfg["decoder_ffn_dim"])

    w.add_layer_norm_eps(1e-5)                       # torch nn.LayerNorm default
    w.add_float32(f"{ARCH}.embedding_scale",
                  math.sqrt(d_model) if scale_emb else 1.0)
    w.add_uint32(f"{ARCH}.position_offset", 2)
    w.add_string(f"{ARCH}.activation", act)
    w.add_bool(f"{ARCH}.pre_layernorm", True)
    w.add_bool(f"{ARCH}.attention.qkv_bias", True)
    w.add_bool(f"{ARCH}.encoder_only", bool(args.encoder_only))

    # ---- tokenizer --------------------------------------------------------
    if not args.no_tokenizer:
        v = load_vocab(md)
        if v is None:
            print("  ! no tokenizer files found - writing weights only",
                  file=sys.stderr)
        else:
            tokens, scores, toktypes, meta = v
            if len(tokens) != n_vocab:
                print(f"  ! vocab {len(tokens)} != config vocab_size {n_vocab} "
                      f"(embedding matrix is padded; expected for NLLB)",
                      file=sys.stderr)
            w.add_tokenizer_model("spm")
            w.add_tokenizer_pre("default")
            w.add_token_list(tokens)
            w.add_token_scores(scores)
            w.add_token_types(toktypes)
            w.add_bos_token_id(bos_id)
            w.add_eos_token_id(eos_id)
            w.add_pad_token_id(pad_id)
            w.add_unk_token_id(cfg.get("unk_token_id", 3))
            w.add_add_bos_token(False)   # NLLB prepends a source LANGUAGE code
            w.add_add_eos_token(True)
            w.add_string(f"tokenizer.{ARCH}.vocab_source", meta["source"])
            print(f"  vocab: {len(tokens)} tokens from {meta['source']}")

    # ---- tensors ----------------------------------------------------------
    n_written = 0

    def emit(name: str, arr: np.ndarray):
        nonlocal n_written
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if args.outtype == "f16" and not should_keep_f32(name):
            arr = arr.astype(np.float16)
        w.add_tensor(name, arr)
        n_written += 1
        if n_written <= 4 or n_written % 25 == 0:
            print(f"  {name:44s} {str(list(arr.shape)):18s} {arr.dtype}")

    embed = ck.first("model.shared.weight",
                     "model.encoder.embed_tokens.weight",
                     "model.decoder.embed_tokens.weight",
                     "shared.weight")
    emit("token_embd.weight", embed)
    emit("position_embd.weight",
         resolve_position_table(ck, max_pos, d_model, pad_id))

    for i in range(cfg["encoder_layers"]):
        src, dst = f"model.encoder.layers.{i}.", f"enc.blk.{i}."
        for hf, gg in ATTN + FFN:
            emit(dst + gg, ck.get(src + hf))
    emit("enc.output_norm.weight", ck.get("model.encoder.layer_norm.weight"))
    emit("enc.output_norm.bias",   ck.get("model.encoder.layer_norm.bias"))

    if not args.encoder_only:
        for i in range(cfg["decoder_layers"]):
            src, dst = f"model.decoder.layers.{i}.", f"dec.blk.{i}."
            for hf, gg in ATTN + CROSS + FFN:
                emit(dst + gg, ck.get(src + hf))
        emit("dec.output_norm.weight", ck.get("model.decoder.layer_norm.weight"))
        emit("dec.output_norm.bias",   ck.get("model.decoder.layer_norm.bias"))

        # lm_head is tied to the shared embedding in NLLB; only emit if untied.
        if "lm_head.weight" in ck:
            lm = ck.get("lm_head.weight")
            if not np.array_equal(lm, embed):
                emit("output.weight", lm)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"\nwrote {out}  ({n_written} tensors, "
          f"{os.path.getsize(out) / 1e9:.2f} GB on disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
