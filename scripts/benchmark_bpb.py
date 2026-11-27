#!/usr/bin/env python3
"""Fixed held-out PT-BR validation builder and tokenizer-independent BPB benchmark.

Two modes:
1. Build a fixed, reproducible held-out PT-BR validation file (~200 KB UTF-8 JSONL) from
   Polygl0t/gigaverbo-v2 (edu_high) using a disjoint seed far from training:
     python scripts/benchmark_bpb.py --build-val val_ptbr_200kb.jsonl --target-bytes 200000

2. Benchmark CinLM checkpoints (including Looped Core at R=1,2,...) and optional Hugging Face
   baselines (e.g. HuggingFaceTB/SmolLM2-135M, google/gemma-3-270m) on the EXACT SAME UTF-8
   validation bytes using tokenizer-independent Bits-Per-Byte (BPB):
     python scripts/benchmark_bpb.py \
       --val-file val_ptbr_200kb.jsonl \
       --cinlm-tokenizer cinlm_bpe_32k.json \
       --cinlm CinLM-v1-Base=/path/to/ckpt_step_10000.pt \
       --hf-model HuggingFaceTB/SmolLM2-135M \
       --out bpb_leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from cinlm.config import CinLMConfig
from cinlm.model import CinLM


def build_heldout_ptbr_val(
    out_path: str | Path,
    target_bytes: int = 200_000,
    seed: int = 99_999,
    skip_docs: int = 25_000,
    min_chars: int = 300,
    max_chars: int = 4000,
) -> Dict[str, Any]:
    """Stream disjoint PT-BR documents from Polygl0t/gigaverbo-v2 into a fixed UTF-8 JSONL."""
    from datasets import load_dataset

    ds = load_dataset("Polygl0t/gigaverbo-v2", "edu_high", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    docs: List[str] = []
    total_bytes = 0
    seen = 0

    for row in ds:
        seen += 1
        if seen <= skip_docs:
            continue
        text = str(row.get("text") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        b = len(text.encode("utf-8"))
        docs.append(text)
        total_bytes += b
        if total_bytes >= target_bytes:
            break

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(docs):
            f.write(json.dumps({"id": idx, "text": text}, ensure_ascii=False) + "\n")

    info = {
        "out_path": str(out),
        "n_docs": len(docs),
        "total_utf8_bytes": total_bytes,
        "seed": seed,
        "skip_docs": skip_docs,
    }
    print(f"[BPB Val] Built fixed held-out PT-BR val set: {json.dumps(info, ensure_ascii=False)}")
    return info


def load_val_texts(val_path: str | Path) -> List[str]:
    lines = Path(val_path).read_text(encoding="utf-8").splitlines()
    texts: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("{"):
            try:
                t = str(json.loads(s).get("text", "")).strip()
                if t:
                    texts.append(t)
                continue
            except Exception:
                pass
        texts.append(s)
    return texts


def _load_cinlm(ckpt_path: str, device: torch.device) -> Tuple[CinLM, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    cfg = CinLMConfig(**cfg_dict)
    model = CinLM(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    meta = {
        "step": int(ckpt.get("step", 0)) + 1,
        "tokens_seen": int(ckpt.get("tokens_seen", 0)),
        "total_params": model.count_parameters()["exact_total"],
        "active_params": cfg.estimate_parameters()["active_per_token"],
        "loop_core_steps": cfg.loop_core_steps,
    }
    return model, meta


@torch.no_grad()
def eval_cinlm_bpb(
    model: CinLM,
    tok: Any,
    texts: List[str],
    device: torch.device,
    num_loops: int | None = None,
) -> Dict[str, Any]:
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    max_len = model.config.max_seq_len

    t0 = time.perf_counter()
    for text in texts:
        ids = tok.encode(text).ids
        if len(ids) < 2:
            continue
        if len(ids) > max_len:
            ids = ids[:max_len]
            text_eval = tok.decode(ids)
        else:
            text_eval = text

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, num_loops=num_loops)["logits"]
        shift_logits = logits[:, :-1, :].float()
        shift_targets = x[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tokens += int(shift_targets.numel())
        total_bytes += len(text_eval.encode("utf-8"))

    dt = time.perf_counter() - t0
    bpb = total_nll / (max(1, total_bytes) * math.log(2.0))
    mean_ce = total_nll / max(1, total_tokens)
    return {
        "n_docs": len(texts),
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_bytes,
        "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
        "mean_ce_nats": round(mean_ce, 5),
        "token_perplexity": round(math.exp(min(mean_ce, 20.0)), 4),
        "bits_per_byte": round(bpb, 5),
        "eval_sec": round(dt, 2),
    }


@torch.no_grad()
def eval_hf_model_bpb(
    model_id: str,
    texts: List[str],
    device: torch.device,
    max_len: int = 2048,
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    t0 = time.perf_counter()
    for text in texts:
        ids = tok.encode(text, add_special_tokens=True)
        if len(ids) < 2:
            continue
        if len(ids) > max_len:
            ids = ids[:max_len]
            text_eval = tok.decode(ids, skip_special_tokens=True)
        else:
            text_eval = text

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=x).logits
        shift_logits = logits[:, :-1, :].float()
        shift_targets = x[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tokens += int(shift_targets.numel())
        total_bytes += len(text_eval.encode("utf-8"))

    dt = time.perf_counter() - t0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bpb = total_nll / (max(1, total_bytes) * math.log(2.0))
    mean_ce = total_nll / max(1, total_tokens)
    return {
        "model": model_id,
        "kind": "hf",
        "total_params": total_params,
        "active_params": total_params,
        "n_docs": len(texts),
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_bytes,
        "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
        "mean_ce_nats": round(mean_ce, 5),
        "token_perplexity": round(math.exp(min(mean_ce, 20.0)), 4),
        "bits_per_byte": round(bpb, 5),
        "eval_sec": round(dt, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed PT-BR Held-Out BPB Benchmark")
    parser.add_argument("--build-val", default=None, help="Build fixed held-out PT-BR JSONL at this path")
    parser.add_argument("--target-bytes", type=int, default=200_000)
    parser.add_argument("--val-seed", type=int, default=99_999)
    parser.add_argument("--skip-docs", type=int, default=25_000)

    parser.add_argument("--val-file", default=None, help="Path to fixed UTF-8 validation file")
    parser.add_argument("--cinlm-tokenizer", default=None, help="Path to CinLM tokenizer.json")
    parser.add_argument(
        "--cinlm",
        action="append",
        default=[],
        help="CinLM checkpoint as label=/path/to/ckpt.pt (repeatable)",
    )
    parser.add_argument(
        "--cinlm-loops",
        default="",
        help="Optional comma-separated loop depths to test for Looped CinLM checkpoints, e.g. '1,2,4'",
    )
    parser.add_argument(
        "--hf-model",
        action="append",
        default=[],
        help="Hugging Face CausalLM model id to compare on identical bytes (repeatable)",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path for leaderboard")
    args = parser.parse_args()

    if args.build_val:
        build_heldout_ptbr_val(
            args.build_val,
            target_bytes=args.target_bytes,
            seed=args.val_seed,
            skip_docs=args.skip_docs,
        )
        if not args.val_file and not args.cinlm and not args.hf_model:
            return
        if not args.val_file:
            args.val_file = args.build_val

    if not args.val_file:
        raise SystemExit("Provide --build-val and/or --val-file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = load_val_texts(args.val_file)
    total_bytes_raw = sum(len(t.encode("utf-8")) for t in texts)
    print(f"[BPB Bench] Loaded {len(texts)} docs ({total_bytes_raw:,} UTF-8 bytes) from {args.val_file}")

    rows: List[Dict[str, Any]] = []
    loop_list = [int(x) for x in args.cinlm_loops.split(",") if x.strip()] if args.cinlm_loops else []

    if args.cinlm:
        from tokenizers import Tokenizer

        assert args.cinlm_tokenizer, "Provide --cinlm-tokenizer when benchmarking --cinlm checkpoints"
        tok = Tokenizer.from_file(args.cinlm_tokenizer)
        for spec in args.cinlm:
            if "=" in spec:
                label, ckpt_path = spec.split("=", 1)
            else:
                label, ckpt_path = Path(spec).stem, spec
            model, meta = _load_cinlm(ckpt_path, device)
            depths = loop_list if (loop_list and model.config.loop_core_steps > 1) else [model.config.loop_core_steps]
            for d in depths:
                if d > model.config.loop_max_steps:
                    continue
                res = eval_cinlm_bpb(model, tok, texts, device, num_loops=d)
                entry = {
                    "model": f"{label} (R={d})" if len(depths) > 1 or d > 1 else label,
                    "kind": "cinlm",
                    "num_loops": d,
                    **meta,
                    **res,
                }
                rows.append(entry)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for hf_id in args.hf_model:
        print(f"[BPB Bench] Evaluating HF baseline: {hf_id} ...")
        rows.append(eval_hf_model_bpb(hf_id, texts, device))

    rows.sort(key=lambda r: r["bits_per_byte"])
    print("\n| Model | Active Params | Bytes/Tok | Token PPL | **BPB (↓)** | Eval (s) |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        act_m = r["active_params"] / 1e6
        print(
            f"| {r['model']} | {act_m:.1f}M | {r['bytes_per_token']:.2f} | "
            f"{r['token_perplexity']:.2f} | **{r['bits_per_byte']:.4f}** | {r['eval_sec']:.1f} |"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved BPB leaderboard to: {args.out}")


if __name__ == "__main__":
    main()
