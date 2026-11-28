#!/usr/bin/env python3
"""CinLM training script — supports from-scratch pretraining & controlled WSD tournament branches.

Features:
- Exact preset configs (smoke / dense_global_100m / dense_hybrid_100m / hybrid_engram_100m / full_260m_a100m)
- Optimizers: fused AdamW or Muon+AuxAdamW (`--optimizer muon`) + separate Engram table LR (`--engram-lr-mult`)
- Schedules: Cosine or WSD (Warmup-Stable-Decay, `--schedule wsd`) for clean checkpoint branching
- Function-preserving branch initialization (`--init-from <ckpt.pt>`) with optional Looped Core (`--loop-core-steps`)
  and Multi-Token Prediction (`--mtp-heads`)
- Deterministic data slicing (`--data-seed`) so competing branches see the exact same token stream
- Iso-token, iso-FLOP (`cum_rel_flops`), and wall-clock (`wall_time_sec`) JSONL telemetry
- Atomic `ckpt_latest.pt` + rotated numbered snapshots + auxiliary-loss-free MoE router bias updates
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from cinlm.config import CinLMConfig, get_preset_config
from cinlm.data import (
    DEFAULT_PTBR_PRETRAIN_SOURCES,
    HIGH_DENSITY_PTBR_PRETRAIN_SOURCES,
    PackedTokenStreamDataset,
)
from cinlm.model import CinLM
from cinlm.muon import build_cinlm_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    schedule: str = "cosine",
    wsd_decay_frac: float = 0.2,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    if schedule == "wsd":
        decay_steps = max(1, int(round(max_steps * wsd_decay_frac)))
        decay_start = max(warmup_steps, max_steps - decay_steps)
        if step < decay_start:
            return max_lr
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    # Standard cosine schedule
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, x.clone()


def _cfg_from_dict(cfg_dict: dict) -> CinLMConfig:
    d = dict(cfg_dict)
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return CinLMConfig(**d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or branch CinLM")
    parser.add_argument(
        "--preset",
        default="full_260m_a100m",
        choices=[
            "smoke",
            "dense_global_100m",
            "dense_hybrid_100m",
            "hybrid_engram_100m",
            "full_260m_a100m",
            "full_1b_a200m",
        ],
    )
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (omit for --synthetic smoke)")
    parser.add_argument(
        "--data-mix",
        default="default",
        choices=["default", "high_density"],
        help="PT-BR pretrain mixture: default (70/15/15) or high_density (55/30/15 synth-boosted)",
    )
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic random tokens (speed/smoke check)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    parser.add_argument("--wsd-decay-frac", type=float, default=0.2, help="Final decay fraction for WSD schedule")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--muon-lr-mult", type=float, default=1.0, help="Muon LR multiplier (1.0 = Moonshot RMS-matched)")
    parser.add_argument("--engram-lr-mult", type=float, default=1.0, help="Separate LR multiplier for sparse Engram tables")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Explicit dataset shuffle seed (defaults to --seed, or 20042 when --init-from is set)",
    )

    # Experimental CinLM v2 architecture extensions (default OFF -> exact v1 compatibility)
    parser.add_argument("--loop-core-steps", type=int, default=1, help="Recursive Core loop depth (1 = off, 2/4 = Looped CinLM)")
    parser.add_argument("--loop-start-layer", type=int, default=None)
    parser.add_argument("--loop-end-layer", type=int, default=None)
    parser.add_argument("--loop-max-steps", type=int, default=4)
    parser.add_argument("--loop-train-mode", default="random", choices=["fixed", "random"])
    parser.add_argument("--mtp-heads", type=int, default=0, help="Multi-Token Prediction auxiliary heads (0 = off, 1 = t+2)")
    parser.add_argument("--mtp-weight", type=float, default=0.3)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable per-layer gradient checkpointing to save VRAM")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=2, help="Max numbered checkpoints to keep (saves Drive space)")
    parser.add_argument("--out-dir", default="runs/cinlm_run")
    parser.add_argument("--init-from", default=None, help="Initialize branch weights from an existing checkpoint (fresh optimizer & step=0)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true", help="Automatically resume from <out-dir>/ckpt_latest.pt if it exists")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        if cc_major >= 8 and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            use_amp = True
            use_scaler = False
        else:
            autocast_dtype = torch.float16
            use_amp = True
            use_scaler = True
            print(
                f"[CinLM] Warning: {gpu_name} (sm_{cc_major}{cc_minor}) does not support native BF16; "
                "falling back to FP16 + GradScaler (and disabling compile if sm<80)."
            )
    else:
        autocast_dtype = torch.float32
        use_amp = False
        use_scaler = False
        cc_major = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    resume_path: Optional[Path] = Path(args.resume) if args.resume else None
    if resume_path is None and args.auto_resume:
        latest = out_dir / "ckpt_latest.pt"
        if latest.exists():
            resume_path = latest

    # Resolve config: load checkpoint on CPU first to avoid doubling peak GPU VRAM on 24GB GPUs (L4)
    init_ckpt = None
    if resume_path is not None and resume_path.exists():
        init_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        cfg = _cfg_from_dict(init_ckpt["config"])
    elif args.init_from:
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = _cfg_from_dict(init_ckpt["config"])
        # Apply requested branch architecture extensions (function-preserving at step 0)
        cfg.loop_core_steps = args.loop_core_steps
        if args.loop_start_layer is not None:
            cfg.loop_start_layer = args.loop_start_layer
        if args.loop_end_layer is not None:
            cfg.loop_end_layer = args.loop_end_layer
        cfg.loop_max_steps = max(args.loop_max_steps, args.loop_core_steps)
        cfg.mtp_num_heads = args.mtp_heads
        cfg.mtp_loss_weight = args.mtp_weight
        cfg.__post_init__()
    else:
        cfg = get_preset_config(args.preset)  # type: ignore[arg-type]
        cfg.loop_core_steps = args.loop_core_steps
        if args.loop_start_layer is not None:
            cfg.loop_start_layer = args.loop_start_layer
        if args.loop_end_layer is not None:
            cfg.loop_end_layer = args.loop_end_layer
        cfg.loop_max_steps = max(args.loop_max_steps, args.loop_core_steps)
        cfg.mtp_num_heads = args.mtp_heads
        cfg.mtp_loss_weight = args.mtp_weight
        cfg.__post_init__()

    if args.seq_len <= cfg.max_seq_len:
        seq_len = args.seq_len
    else:
        cfg.max_seq_len = args.seq_len
        seq_len = args.seq_len

    model = CinLM(cfg).to(device)
    model.gradient_checkpointing = args.grad_checkpoint
    param_info = model.count_parameters()

    optimizer, opt_counts = build_cinlm_optimizer(
        model,
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr_mult=args.muon_lr_mult,
        engram_lr_mult=args.engram_lr_mult,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    tokens_seen = 0
    base_tokens_seen = 0
    cum_rel_flops = 0.0
    wall_offset_sec = 0.0

    if resume_path is not None and resume_path.exists() and init_ckpt is not None:
        model.load_state_dict(init_ckpt["model"])
        optimizer.load_state_dict(init_ckpt["optimizer"])
        start_step = int(init_ckpt["step"]) + 1
        tokens_seen = int(init_ckpt["tokens_seen"])
        base_tokens_seen = int(init_ckpt.get("base_tokens_seen", 0))
        cum_rel_flops = float(init_ckpt.get("cum_rel_flops", 0.0))
        wall_offset_sec = float(init_ckpt.get("wall_time_sec", 0.0))
        print(f"[CinLM] Resumed from {resume_path} at step {start_step} ({tokens_seen:,} total tokens)")
    elif args.init_from and init_ckpt is not None:
        has_extensions = cfg.loop_core_steps > 1 or cfg.mtp_num_heads > 0
        load_res = model.load_state_dict(init_ckpt["model"], strict=not has_extensions)
        assert not load_res.unexpected_keys, f"Unexpected keys when loading --init-from: {load_res.unexpected_keys}"
        allowed_prefixes = ("loop_", "mtp_")
        bad_missing = [k for k in load_res.missing_keys if not k.startswith(allowed_prefixes)]
        assert not bad_missing, f"Missing non-extension keys when loading --init-from: {bad_missing}"
        base_tokens_seen = int(init_ckpt.get("tokens_seen", 0))
        tokens_seen = base_tokens_seen
        print(
            f"[CinLM] Branched from {args.init_from} (base_tokens={base_tokens_seen:,}, "
            f"new_extension_keys={len(load_res.missing_keys)})"
        )

    if init_ckpt is not None:
        del init_ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"[CinLM] model={cfg.name} opt={args.optimizer} sched={args.schedule} "
        f"loop_R={cfg.loop_core_steps}({args.loop_train_mode}) mtp={cfg.mtp_num_heads} "
        f"engram_lr_mult={args.engram_lr_mult} device={device} dtype={autocast_dtype}"
    )
    print(
        f"[CinLM] total={param_info['exact_total']:,} active/tok={param_info['estimated_active_per_token']:,} "
        f"opt_split={opt_counts}"
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    enable_compile = args.compile and device.type == "cuda" and cc_major >= 8
    run_model = torch.compile(model) if enable_compile else model

    base_data_seed = args.data_seed if args.data_seed is not None else (20042 if args.init_from else args.seed)
    data_iter: Optional[object] = None
    if not args.synthetic:
        assert args.tokenizer is not None, "Provide --tokenizer path or pass --synthetic"
        sources = (
            HIGH_DENSITY_PTBR_PRETRAIN_SOURCES
            if args.data_mix == "high_density"
            else DEFAULT_PTBR_PRETRAIN_SOURCES
        )
        ds = PackedTokenStreamDataset(
            args.tokenizer, sources=sources, seq_len=seq_len, seed=base_data_seed + start_step
        )
        n_workers = min(4, max(2, (os.cpu_count() or 2) // 2)) if device.type == "cuda" else 0
        loader_kwargs = (
            {"num_workers": n_workers, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True}
            if n_workers > 0
            else {}
        )
        loader = DataLoader(ds, batch_size=args.batch_size, **loader_kwargs)
        data_iter = iter(loader)

    run_model.train()
    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    core_frac = (cfg.loop_end_layer - cfg.loop_start_layer) / max(1, cfg.n_layers)
    loop_rng = random.Random(args.seed + 777 + start_step)

    run_start_t = time.perf_counter()
    t0 = run_start_t

    for step in range(start_step, args.max_steps):
        lr = get_lr(
            step,
            args.warmup_steps,
            args.max_steps,
            args.lr,
            args.min_lr,
            schedule=args.schedule,
            wsd_decay_frac=args.wsd_decay_frac,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr * float(pg.get("lr_mult", 1.0))

        step_loops = (
            loop_rng.randint(1, cfg.loop_core_steps)
            if (cfg.loop_core_steps > 1 and args.loop_train_mode == "random")
            else cfg.loop_core_steps
        )
        step_flops_mult = 1.0 + max(0, step_loops - 1) * core_frac

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        ntp_accum = 0.0
        mtp_accum = 0.0

        for _ in range(args.grad_accum):
            if args.synthetic:
                x, y = synthetic_batch(cfg.vocab_size, args.batch_size, seq_len, device)
            else:
                x, y = next(data_iter)  # type: ignore[arg-type]
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                out = run_model(x, labels=y, num_loops=step_loops)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            loss_accum += float(loss.detach().item())
            if out.get("ntp_loss") is not None:
                ntp_accum += float(out["ntp_loss"].detach().item()) / args.grad_accum
            if out.get("mtp_loss") is not None:
                mtp_accum += float(out["mtp_loss"].detach().item()) / args.grad_accum

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        scaler.step(optimizer)
        scaler.update()
        model.update_moe_router_biases()

        tokens_seen += tokens_per_step
        cum_rel_flops += step_flops_mult * tokens_per_step

        if (step + 1) % args.log_every == 0 or step == start_step:
            now = time.perf_counter()
            dt = now - t0
            tok_s = (tokens_per_step * (args.log_every if step > start_step else 1)) / max(dt, 1e-6)
            rec = {
                "step": step + 1,
                "loss": round(loss_accum, 5),
                "ntp_loss": round(ntp_accum, 5),
                "lr": round(lr, 7),
                "grad_norm": round(grad_norm, 4),
                "tokens_seen": tokens_seen,
                "branch_tokens": tokens_seen - base_tokens_seen,
                "tok_per_sec": round(tok_s, 1),
                "wall_time_sec": round(wall_offset_sec + (now - run_start_t), 1),
                "cum_rel_flops": round(cum_rel_flops / 1e6, 2),
            }
            if cfg.mtp_num_heads > 0:
                rec["mtp_loss"] = round(mtp_accum, 5)
            if cfg.loop_core_steps > 1:
                rec["step_loops"] = step_loops
            print(json.dumps(rec))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            t0 = time.perf_counter()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            wall_now = wall_offset_sec + (time.perf_counter() - run_start_t)
            payload = {
                "config": asdict(cfg),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "tokens_seen": tokens_seen,
                "base_tokens_seen": base_tokens_seen,
                "cum_rel_flops": cum_rel_flops,
                "wall_time_sec": wall_now,
            }
            latest_tmp = out_dir / "ckpt_latest.pt.tmp"
            latest_path = out_dir / "ckpt_latest.pt"
            torch.save(payload, latest_tmp)
            os.replace(latest_tmp, latest_path)

            ckpt_path = out_dir / f"ckpt_step_{step + 1}.pt"
            torch.save(payload, ckpt_path)
            if args.keep_checkpoints > 0:
                numbered = sorted(
                    out_dir.glob("ckpt_step_*.pt"),
                    key=lambda p: int(p.stem.split("_")[-1]),
                )
                for old in numbered[: -args.keep_checkpoints]:
                    old.unlink(missing_ok=True)
            print(f"[CinLM] Saved checkpoint: {ckpt_path} (+ atomic ckpt_latest.pt)")
            t0 = time.perf_counter()


if __name__ == "__main__":
    main()
