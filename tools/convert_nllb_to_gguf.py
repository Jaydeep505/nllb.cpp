#!/usr/bin/env python3
"""
Convert facebook/nllb-200-distilled-600M (HF M2M100 architecture) to GGUF.

Reads either model.safetensors or pytorch_model.bin (sharded or not). The
safetensors path parses the container itself - no safetensors package, no
torch, and f32/f16/**bf16** all work, which matters because bf16 is what most
fine-tuning runs save and is the one dtype `safe_open(framework="np")` cannot
return. The .bin path imports torch lazily, only if a .bin is what you have.

    pip install numpy gguf sentencepiece regex
    python tools/convert_nllb_to_gguf.py models/nllb-600M \
        --outfile models/nllb-600M-f32.gguf --outtype f32

Fine-tunes: everything here comes from config.json, so any M2M100-shaped
checkpoint converts at whatever depth and width it has. A fine-tune that also
TRIMMED or REORDERED the vocabulary breaks the merge derivation described under
load_vocab(); the converter detects that and embeds the merge list instead of
refusing. See TOKENIZER.md.

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
import base64
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

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

ST_DTYPES = {
    "F64":  np.dtype("<f8"),
    "F32":  np.dtype("<f4"),
    "F16":  np.dtype("<f2"),
    "I64":  np.dtype("<i8"),
    "I32":  np.dtype("<i4"),
    "I16":  np.dtype("<i2"),
    "I8":   np.dtype("i1"),
    "U8":   np.dtype("u1"),
    "BOOL": np.dtype("?"),
    # BF16 has no numpy equivalent; we widen it by hand. See _st_tensor().
    "BF16": None,
}


def read_st_header(path: Path) -> tuple[dict, int]:
    """(header dict, byte offset of the data segment) of a .safetensors file.

    The format is fixed: u64le header length, that many bytes of JSON, then the
    tensor data. We parse it ourselves rather than going through
    `safetensors.safe_open(framework="np")` because that call raises
    `TypeError: data type 'bfloat16' not understood` on any BF16 tensor - and
    BF16 is what most fine-tuning runs save. Reading the header directly costs
    us nothing and drops the safetensors dependency entirely.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if n <= 0 or n > 256 << 20:
            raise ValueError(f"{path.name}: implausible safetensors header ({n} bytes)")
        header = json.loads(f.read(n))
    return header, 8 + n


class Checkpoint:
    """Uniform read-only view over an HF checkpoint directory.

    Prefers safetensors (mmap'd, no torch, BF16 included). Falls back to
    pytorch_model.bin, which is a pickle and therefore needs torch to unpickle.
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
        self.backend = "safetensors"

        if index.exists():
            with open(index) as f:
                for name, shard in json.load(f)["weight_map"].items():
                    self.map[name] = self.dir / shard
            return
        for path in ([single] if single.exists() else shards):
            header, _ = read_st_header(path)
            for name in header:
                if name != "__metadata__":
                    self.map[name] = path

    def _st_tensor(self, path: Path, name: str) -> np.ndarray:
        """One tensor out of a safetensors shard, as float-ready numpy.

        BF16 is widened to f32 exactly: a bfloat16 IS the top 16 bits of the
        f32 with the same value, so shifting left by 16 is lossless and needs
        no rounding decisions. F8 we refuse rather than guess at.
        """
        if path not in self._handles:
            header, data_off = read_st_header(path)
            self._handles[path] = (header, np.memmap(path, dtype=np.uint8, mode="r"),
                                   data_off)
        header, mm, data_off = self._handles[path]

        info = header[name]
        st_dtype = info["dtype"]
        if st_dtype not in ST_DTYPES:
            raise TypeError(
                f"{name}: safetensors dtype {st_dtype!r} is not supported by "
                f"this converter. Re-save the checkpoint in f32, f16 or bf16.")
        beg, end = (data_off + o for o in info["data_offsets"])
        raw = mm[beg:end]

        if st_dtype == "BF16":
            u16 = raw.view("<u2")
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = raw.view(ST_DTYPES[st_dtype])
        return arr.reshape(info["shape"])

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
            return self._st_tensor(self.map[name], name)

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

def fnv1a64(items: Iterable[str]) -> int:
    """FNV-1a over `\\n`-joined UTF-8. Must match nllb_vocab's C++ copy."""
    h = 0xCBF29CE484222325
    for i, s in enumerate(items):
        if i:
            h = ((h ^ 0x0A) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        for b in s.encode("utf-8"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def derive_merges(tokens: Sequence[str]) -> list[str]:
    """Rebuild HF's BPE merge list from the vocabulary ORDER alone.

    This is transformers' SentencePieceExtractor.extract() with
    vocab_scores=None, which is the branch that produced NLLB's tokenizer.json:
    for each piece in id order, every split whose halves are both in the vocab,
    those splits ordered by (left id, right id).

    Verified byte-for-byte against the 514098 merges in the real
    facebook/nllb-200-distilled-600M tokenizer.json - which is why the C++ side
    can derive them too instead of the GGUF carrying ~6 MB of merge strings.
    `tokenizer.nllb.merges_hash` is what keeps that a checked invariant rather
    than an assumption: see the comment where it is written.
    """
    index = {t: i for i, t in enumerate(tokens)}
    out: list[str] = []
    for piece in tokens:
        local = []
        for i in range(1, len(piece)):
            l, r = piece[:i], piece[i:]
            il, ir = index.get(l), index.get(r)
            if il is not None and ir is not None:
                local.append((il, ir, l, r))
        local.sort(key=lambda x: (x[0], x[1]))
        out.extend(f"{l} {r}" for _, _, l, r in local)
    return out


def grapheme_tables():
    """(gcb_flat, extpict_flat, ucd_version), or None if `regex` is missing.

    These are the UAX#29 tables the normalizer needs to find grapheme cluster
    boundaries. They are NOT a property of the model - every SentencePiece
    checkpoint would carry the same ~8 KB - but they ARE a property of the ids,
    because a cluster boundary that moves moves the tokenization. Baking them
    in makes the ids a function of the GGUF alone rather than of the GGUF and
    whichever Unicode release the binary happened to be built against.

    Computed through tools/gen_grapheme_tables.py, the same code that writes
    the committed src/nllb-grapheme-data.h, so the two cannot disagree.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from gen_grapheme_tables import build_tables, flatten
        rows, pr, ucd = build_tables()
    except ImportError as e:
        print(f"  ! no grapheme tables in this file ({e}); the C++ will fall "
              "back to its compiled-in copy", file=sys.stderr)
        return None
    gcb, ext = flatten(rows, pr)
    return gcb, ext, ucd


def load_vocab(model_dir: Path):
    """Return (tokens, scores, toktypes, meta) or None.

    Prefers tokenizer.json, whose vocab dict already has NLLB's fairseq +1 id
    offset and the 200+ FLORES language codes baked in - so we never have to
    guess at the offset arithmetic.

    meta carries two things the tokenizer needs that the old schema had no room
    for: `charsmap` (SentencePiece's precompiled normalizer, without which
    U+FB01 never becomes "fi" and U+2044 never appears) and `merges_hash`.

    NOTE: NLLB's tokenizer is BPE, not Unigram, despite the Unigram-shaped
    `scores` array this converter has always written. `sentencepiece.bpe.model`
    means what its name says: model_type == BPE, and the scores are negated
    merge ranks rather than log-probabilities. Nothing reads them today.

    `merges_list`, when present, means the merge derivation does not apply to
    this vocabulary and main() must write the list into the GGUF.
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

        meta = {"source": "tokenizer.json", "model_type": tj["model"]["type"]}

        # HF's AddedVocabulary cuts these out of the raw input before the
        # normalizer ever sees it, so "x eng_Latn y" is three chunks and not
        # one sentence. The C++ needs the strings AND their flags.
        # bit0 single_word, bit1 lstrip, bit2 rstrip, bit3 normalized.
        atoks, aflags = [], []
        for t in tj.get("added_tokens", []):
            if t.get("single_word"):
                raise SystemExit(
                    f"added token {t['content']!r} sets single_word, which "
                    "nllb_vocab does not implement. Refusing rather than "
                    "writing a GGUF that tokenizes differently to HF.")
            atoks.append(t["content"])
            aflags.append((2 if t.get("lstrip") else 0) |
                          (4 if t.get("rstrip") else 0) |
                          (8 if t.get("normalized") else 0))
        meta["added_tokens"] = atoks
        meta["added_flags"] = aflags

        # the normalizer. HF serialises it as a Sequence; we want the
        # Precompiled member's charsmap, and we assert the rest of the sequence
        # is the ` {2,}` -> ` ` Replace that nllb_vocab hard-codes. A different
        # normalizer stack here means the C++ is normalising something else.
        norm = tj.get("normalizer") or {}
        seq = norm.get("normalizers", [norm])
        kinds = [x.get("type") for x in seq]
        if kinds != ["Precompiled", "Replace"]:
            raise SystemExit(
                f"unexpected normalizer stack {kinds}; nllb_vocab implements "
                "[Precompiled, Replace(' {2,}' -> ' ')] and nothing else")
        rep = seq[1]
        if rep.get("pattern", {}).get("Regex") != " {2,}" or rep.get("content") != " ":
            raise SystemExit(f"unexpected Replace rule: {rep}")
        meta["charsmap"] = base64.b64decode(seq[0]["precompiled_charsmap"])

        pre = tj.get("pre_tokenizer") or {}
        if pre.get("type") != "Metaspace" or pre.get("replacement") != "▁":
            raise SystemExit(f"unexpected pre_tokenizer: {pre}")

        # The merge list. We do NOT write it: it is exactly reproducible from
        # the vocabulary order, and the C++ rebuilds it the same way. What we
        # write is a hash of the merges HF actually serialised, so that
        # derivation is checked rather than trusted - if a future checkpoint's
        # tokenizer.json was built by a transformers version whose extractor
        # orders splits differently, the C++ refuses to load instead of
        # tokenizing subtly differently for the rest of its life.
        if tj["model"]["type"] != "BPE":
            raise SystemExit(
                f"tokenizer.json is a {tj['model']['type']} model; nllb_vocab "
                "implements BPE only.")
        hf_merges = [m if isinstance(m, str) else " ".join(m)
                     for m in tj["model"]["merges"]]
        meta["merges_hash"] = fnv1a64(hf_merges)
        meta["n_merges"] = len(hf_merges)

        meta["merges"] = hf_merges          # only WRITTEN if --embed-merges

        derived = derive_merges(tokens)
        if derived == hf_merges:
            meta["merges_hash_source"] = "tokenizer.json"
        else:
            # A vocabulary that was trimmed or extended after training - which
            # is what most language-specific fine-tunes do - reindexes the
            # pieces, and HF keeps the ORIGINAL merge order filtered rather
            # than re-deriving. The derivation is then simply not applicable to
            # this checkpoint, which is a different thing from the checkpoint
            # being broken. So carry the list (~6 MB on a full vocab, less on a
            # trimmed one) and let the C++ use it verbatim; merges_hash still
            # covers it, it just now checks a transfer rather than a
            # derivation.
            print(f"  ! derivation does not apply to this vocabulary "
                  f"({len(derived)} derived vs {len(hf_merges)} in "
                  f"tokenizer.json); embedding the merge list instead",
                  file=sys.stderr)
            meta["merges_list"] = hf_merges
            meta["merges_hash_source"] = "tokenizer.json (embedded list)"
        return tokens, scores, toktypes, meta

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

    # The normalizer lives in the proto, not in the piece list.
    charsmap = b""
    try:
        from sentencepiece import sentencepiece_model_pb2 as spm_pb2
        proto = spm_pb2.ModelProto()
        proto.ParseFromString(sp.serialized_model_proto())
        charsmap = proto.normalizer_spec.precompiled_charsmap
        if proto.trainer_spec.model_type != spm_pb2.TrainerSpec.BPE:
            print("  ! this sentencepiece model is not BPE; the C++ tokenizer "
                  "implements BPE only", file=sys.stderr)
    except Exception as e:                       # noqa: BLE001
        print(f"  ! could not read the normalizer out of the spm proto ({e}); "
              "the C++ tokenizer will refuse to load this GGUF",
              file=sys.stderr)

    # No tokenizer.json means nothing to check the derivation AGAINST: this
    # hash is self-consistent only, and says so. Convert from a snapshot that
    # has tokenizer.json if you want the checked version.
    print("  ! no tokenizer.json: merges are derived and UNVERIFIED",
          file=sys.stderr)
    derived = derive_merges(tokens)
    # Every special here is plain: no lstrip/rstrip/single_word, matched on the
    # raw text. <mask> is the one exception and it is flagged as HF flags it.
    specials = ["<s>", "<pad>", "</s>", "<unk>"] + langs + ["<mask>"]
    return tokens, scores, toktypes, {
        "source": "sentencepiece.bpe.model",
        "charsmap": charsmap,
        "merges": derived,
        "merges_hash": fnv1a64(derived),
        "merges_hash_source": "derived (unverified)",
        "n_merges": len(derived),
        "added_tokens": specials,
        "added_flags": [(2 | 8) if s == "<mask>" else 0 for s in specials],
    }


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
    ap.add_argument("--embed-merges", action="store_true",
                    help="write the merge list into the GGUF even when the C++ "
                         "could derive it from the vocab order (~6 MB, saves "
                         "~0.4 s of load time). A vocabulary the derivation "
                         "does not fit embeds the list regardless.")
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
        # NLLB starts the decoder with EOS and then forces the target language
        # code, so this is 2 and not bos. Write it so the runtime never has to
        # guess - guessing bos here produces fluent output in the wrong
        # language, which is the kind of bug that survives a test suite.
        w.add_decoder_start_token_id(cfg.get("decoder_start_token_id", eos_id))

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

            # --- what the C++ tokenizer actually needs ---------------------
            # SentencePiece's normalizer, as a double-array trie plus a blob of
            # replacement strings. Without it "ﬁ" never becomes "fi", "½" never
            # becomes "1⁄2", NBSP never becomes a space, and the ids diverge
            # from HF on any input that is not plain ASCII.
            cm = meta.get("charsmap") or b""
            if cm:
                w.add_precompiled_charsmap(cm)
                print(f"  charsmap: {len(cm)} bytes")
            else:
                print("  ! NO precompiled_charsmap written - the C++ tokenizer "
                      "will refuse to load this file", file=sys.stderr)

            w.add_array(f"tokenizer.{ARCH}.added_tokens", meta["added_tokens"])
            w.add_array(f"tokenizer.{ARCH}.added_flags", meta["added_flags"])

            # UAX#29 grapheme cluster tables. See grapheme_tables().
            gt = grapheme_tables()
            if gt:
                gcb, ext, ucd = gt
                w.add_array(f"tokenizer.{ARCH}.gcb_ranges", gcb)
                w.add_array(f"tokenizer.{ARCH}.extpict_ranges", ext)
                w.add_string(f"tokenizer.{ARCH}.ucd_version", ucd)
                print(f"  graphemes: {len(gcb) // 3} GCB ranges + "
                      f"{len(ext) // 2} pictographic, UCD {ucd} "
                      f"({(len(gcb) + len(ext)) * 4} bytes)")

            # The merges are DERIVED on the C++ side from the vocab order; this
            # is the hash that makes that derivation a checked invariant.
            # When the derivation does not reproduce this checkpoint's list -
            # or when --embed-merges says so - the list itself goes in the file
            # and the C++ prefers it, with the same hash checking the transfer.
            merges_list = meta.get("merges_list")
            if merges_list is None and args.embed_merges and "merges" in meta:
                merges_list = meta["merges"]
            if merges_list is not None:
                w.add_array(f"tokenizer.{ARCH}.merges", merges_list)
                if not meta.get("merges_hash_source", "").endswith("list)"):
                    meta["merges_hash_source"] = \
                        meta.get("merges_hash_source", "tokenizer.json") + \
                        " (embedded list)"
                print(f"  merges: {len(merges_list)} embedded "
                      f"({sum(len(m) + 1 for m in merges_list) / 1e6:.1f} MB)")
            w.add_uint64(f"tokenizer.{ARCH}.merges_hash", meta["merges_hash"])
            w.add_uint32(f"tokenizer.{ARCH}.n_merges", meta["n_merges"])
            w.add_string(f"tokenizer.{ARCH}.merges_hash_source",
                         meta.get("merges_hash_source", "tokenizer.json"))

            print(f"  vocab: {len(tokens)} tokens from {meta['source']}, "
                  f"{meta['n_merges']} merges "
                  f"(hash {meta['merges_hash']:#018x}, "
                  f"{meta.get('merges_hash_source', 'tokenizer.json')})")

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

    # The GGUF schema has exactly one token_embd.weight, which is only sound
    # while the encoder and the decoder genuinely share it. NLLB does - HF ties
    # both to model.shared - but a checkpoint saved with
    # tie_word_embeddings=False writes all of them out separately, and if they
    # ever differ we would silently encode with the wrong matrix. Nothing
    # downstream can detect that: the shapes match and the numbers look
    # plausible. So check here, where it is still cheap.
    for other in ("model.encoder.embed_tokens.weight",
                  "model.decoder.embed_tokens.weight"):
        if other in ck and not np.array_equal(ck.get(other), embed):
            sys.exit(f"  ! {other} differs from the shared embedding. The GGUF "
                     f"schema stores one token_embd.weight for both towers, so "
                     f"this checkpoint cannot be converted faithfully. "
                     f"(NLLB ties these; if you hit this on a real model the "
                     f"schema needs enc/dec embeddings split, not a workaround.)")
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
