#!/usr/bin/env python3
"""
Dump HF *decoder* states as golden files, teacher-forced, for the C++ diff.

    python tools/dump_reference.py         models/nllb-600M --out ref/ --per-layer
    python tools/dump_decoder_reference.py models/nllb-600M --out ref/ \
        --tgt-lang deu_Latn --per-layer --greedy

Run the encoder dumper FIRST. This script reads `ref/<name>.ids.bin` back and
feeds those exact source ids to the encoder, so the encoder output the decoder
cross-attends to is bit-identical to the one the encoder test already proves.
Regenerating source ids here instead would let the two drift apart silently.

Per case, raw little-endian:

    ref/<name>.dec_ids.bin        int32[Td]              decoder_input_ids
    ref/<name>.dec.bin            float32[Td, d_model]   final decoder output
    ref/<name>.dec_layer_NN.bin   float32[Td, d_model]   with --per-layer
    ref/<name>.logits_last.bin    float32[vocab]         logits at the last step
    ref/<name>.gen.bin            int32[Tg]              with --greedy
    ref/<name>.dec.json           ids, tokens, shapes, checksums
    ref/manifest_dec.txt          "<name> <Td> <d_model> <n_dec_hs> <vocab>"

Layer indexing is the same trap as the encoder, verified against this exact
transformers version rather than assumed:

    dec_layer_00            embeddings (scale + positions) = INPUT to block 0
    dec_layer_i             input to block i = output of block i-1
    dec_layer_NN            output of the last block AFTER dec.output_norm
                            (== dec.bin; feeding it through the tied lm_head
                            reproduces model.logits exactly)

TEACHER FORCING. decoder_input_ids are the *target* sequence shifted right:
[decoder_start_token_id] + target[:-1]. For NLLB the target from the tokenizer
is [tgt_lang_code, ...pieces..., eos] and decoder_start_token_id is eos (2), so
the decoder input begins [2, tgt_lang_code, ...] - exactly what generate()
feeds at step 0 with forced_bos_token_id.

The target text does not have to be a correct translation. Teacher forcing only
needs *valid* ids; correctness of the graph is what is under test, not
translation quality. We tokenise the source text under --tgt-lang, which keeps
the 9 cases multilingual and in sync with the encoder side.

--random-ids skips the tokenizer entirely (deterministic pseudo-random target
ids), which is how you generate goldens for a model with no tokenizer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration

from dump_reference import PROMPTS


def read_i32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="facebook/nllb-200-distilled-600M",
                    help="local snapshot dir, or a hub id")
    ap.add_argument("--out", type=Path, default=Path("ref"))
    ap.add_argument("--tgt-lang", default="deu_Latn")
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--random-ids", action="store_true",
                    help="skip the tokenizer; use deterministic pseudo-random ids")
    ap.add_argument("--greedy", action="store_true",
                    help="also dump generate(num_beams=1) ids for the next milestone")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--all-logits", action="store_true",
                    help="dump [Td, vocab] instead of just the last step "
                         "(~1 MB per token on the 600M - off by default)")
    args = ap.parse_args()

    if not (args.out / "manifest.txt").exists():
        raise SystemExit(
            f"no {args.out}/manifest.txt - run tools/dump_reference.py first; "
            "this script reuses its source ids so encoder and decoder goldens "
            "cannot drift apart")

    model = M2M100ForConditionalGeneration.from_pretrained(args.model,
                                                           dtype=torch.float32)
    model.eval()
    cfg = model.config
    dec_start = cfg.decoder_start_token_id
    print(f"d_model={cfg.d_model} dec_layers={cfg.decoder_layers} "
          f"heads={cfg.decoder_attention_heads} vocab={cfg.vocab_size}")
    print(f"decoder_start_token_id={dec_start} pad={cfg.pad_token_id} "
          f"eos={cfg.eos_token_id}")

    tok = None
    tgt_lang_id = None
    if not args.random_ids:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, tgt_lang=args.tgt_lang)
        tgt_lang_id = tok.convert_tokens_to_ids(args.tgt_lang)
        print(f"tgt_lang={args.tgt_lang} -> id {tgt_lang_id}")

    rng = np.random.default_rng(20240820)
    manifest = {
        "model": str(args.model),
        "tgt_lang": args.tgt_lang if tok else None,
        "tgt_lang_id": int(tgt_lang_id) if tgt_lang_id is not None else None,
        "d_model": cfg.d_model,
        "decoder_layers": cfg.decoder_layers,
        "decoder_attention_heads": cfg.decoder_attention_heads,
        "decoder_start_token_id": int(dec_start),
        "vocab_size": cfg.vocab_size,
        "per_layer": bool(args.per_layer),
        "teacher_forced": True,
        "cases": [],
    }
    lines = []

    for name, text in PROMPTS:
        src_path = args.out / f"{name}.ids.bin"
        if not src_path.exists():
            print(f"  {name:20s} SKIP (no {src_path.name})")
            continue
        src = read_i32(src_path).astype(np.int64)
        src_t = torch.tensor(src[None, :], dtype=torch.long)
        src_am = torch.tensor((src != cfg.pad_token_id).astype(np.int64)[None, :])

        # ---- target ids -> teacher-forced decoder inputs ----
        if tok is not None:
            labels = np.asarray(tok(text_target=text)["input_ids"], dtype=np.int64)
        else:
            n = max(1, len(text.split()) + 1)
            labels = rng.integers(4, cfg.vocab_size, size=n).astype(np.int64)
            labels[-1] = cfg.eos_token_id
        # shift right: [decoder_start] + labels[:-1]  (HF shift_tokens_right)
        dec_ids = np.concatenate([[dec_start], labels[:-1]]).astype(np.int64)
        dec_t = torch.tensor(dec_ids[None, :], dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids=src_t, attention_mask=src_am,
                        decoder_input_ids=dec_t,
                        output_hidden_states=args.per_layer, use_cache=False)

        h = np.ascontiguousarray(
            out.decoder_hidden_states[-1][0].numpy().astype(np.float32)
            if args.per_layer else
            out.last_hidden_state[0].numpy().astype(np.float32))
        logits = out.logits[0].numpy().astype(np.float32)
        Td = int(h.shape[0])

        (args.out / f"{name}.dec_ids.bin").write_bytes(
            np.ascontiguousarray(dec_ids.astype(np.int32)).tobytes())
        (args.out / f"{name}.dec.bin").write_bytes(np.ascontiguousarray(h).tobytes())

        if args.all_logits:
            (args.out / f"{name}.logits.bin").write_bytes(
                np.ascontiguousarray(logits).tobytes())
        (args.out / f"{name}.logits_last.bin").write_bytes(
            np.ascontiguousarray(logits[-1]).tobytes())

        case = {
            "name": name, "text": text,
            "src_ids": src.astype(int).tolist(),
            "dec_ids": dec_ids.astype(int).tolist(),
            "dec_tokens": tok.convert_ids_to_tokens(dec_ids.tolist()) if tok else None,
            "shape": [Td, int(h.shape[1])],
            "dec_sha256": hashlib.sha256(h.tobytes()).hexdigest()[:16],
            "dec_mean": float(h.mean()), "dec_std": float(h.std()),
            "dec_absmax": float(np.abs(h).max()),
            "logits_last_argmax": int(logits[-1].argmax()),
            "logits_last_max": float(logits[-1].max()),
        }

        n_hs = 0
        if args.per_layer:
            for j, hs in enumerate(out.decoder_hidden_states):
                a = np.ascontiguousarray(hs[0].numpy().astype(np.float32))
                (args.out / f"{name}.dec_layer_{j:02d}.bin").write_bytes(a.tobytes())
            n_hs = len(out.decoder_hidden_states)
            case["n_decoder_hidden_states"] = n_hs
            assert n_hs == cfg.decoder_layers + 1, (
                f"expected {cfg.decoder_layers + 1} decoder hidden states, got {n_hs} "
                "- the C++ bisect indexing assumes n_layer+1")

        if args.greedy:
            kw = {}
            if tgt_lang_id is not None:
                kw["forced_bos_token_id"] = int(tgt_lang_id)
            with torch.no_grad():
                g = model.generate(input_ids=src_t, attention_mask=src_am,
                                   num_beams=1, do_sample=False,
                                   max_new_tokens=args.max_new_tokens, **kw)
            gen = np.ascontiguousarray(g[0].numpy().astype(np.int32))
            (args.out / f"{name}.gen.bin").write_bytes(gen.tobytes())
            case["greedy_ids"] = gen.astype(int).tolist()
            case["greedy_text"] = (tok.decode(gen, skip_special_tokens=True)
                                   if tok else None)

        with open(args.out / f"{name}.dec.json", "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)
        manifest["cases"].append(case)
        lines.append(f"{name} {Td} {h.shape[1]} {n_hs} {cfg.vocab_size}")
        print(f"  {name:20s} Td={Td:<3d} std={h.std():.4f} "
              f"argmax={case['logits_last_argmax']}"
              f"{'  +' + str(n_hs) + ' layers' if n_hs else ''}")

    (args.out / "manifest_dec.txt").write_text("\n".join(lines) + "\n")
    with open(args.out / "manifest_dec.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(manifest['cases'])} decoder cases to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
