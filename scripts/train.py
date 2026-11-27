#!/usr/bin/env python3
"""CinLM training script — engineered for Colab G4 (RTX PRO 6000 Blackwell) & reproducible ablations.

Features:
- Deterministic seeding & exact preset configs (smoke / dense_global_100m / dense_hybrid_100m / hybrid_engram_100m / full_260m_a100m)
- BF16 mixed precision + torch.compile option
- Cosine LR schedule with linear warmup
- Auxiliary-loss-free MoE router bias updates (DeepSeek-V3/V4 style)
- Resumable checkpoints with full config + optimizer + step + tokens_seen
- Structured JSONL metric logs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from cinlm.config import PresetName, get_preset_config
from cinlm.data import PackedTokenStreamDataset
from cinlm.model import CinLM


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: CinLM, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # Do not weight-decay 1D norms/biases or Engram lookup tables
        if p.ndim < 2 or "tables" in name or "norm" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )


def synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, x.clone()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CinLM")
    parser.add_argument(
        "--preset",
        default="full_260m_a100m",
        choices=[
            "smoke",
            "dense_global_100m",
            "dense_hybrid_100m",
            "hybrid_engram_100m",
            "full_260m_a100m",
        ],
    )
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (omit for --synthetic smoke)")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic random tokens (speed/smoke check)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable per-layer gradient checkpointing to save VRAM")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=2, help="Max numbered checkpoints to keep (saves Drive space)")
    parser.add_argument("--out-dir", default="runs/cinlm_run")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true", help="Automatically resume from <out-dir>/ckpt_latest.pt if it exists")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only use BF16 on native sm>=80 GPUs (A100/L4/G4/RTX30+); use FP16 + GradScaler on Turing (T4)
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
                "falling back to FP16 + GradScaler (and disabling compile if sm<80). "
                "For full speed, switch Colab runtime to G4/A100."
            )
    else:
        autocast_dtype = torch.float32
        use_amp = False
        use_scaler = False
        cc_major = 0

    cfg = get_preset_config(args.preset)  # type: ignore[arg-type]
    if args.seq_len <= cfg.max_seq_len:
        seq_len = args.seq_len
    else:
        cfg.max_seq_len = args.seq_len
        seq_len = args.seq_len

    model = CinLM(cfg).to(device)
    model.gradient_checkpointing = args.grad_checkpoint
    param_info = model.count_parameters()
    print(f"[CinLM] preset={args.preset} device={device} dtype={autocast_dtype} grad_ckpt={args.grad_checkpoint}")
    print(
        f"[CinLM] total={param_info['exact_total']:,} "
        f"active/tok={param_info['estimated_active_per_token']:,}"
    )

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start_step = 0
    tokens_seen = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    resume_path: Optional[Path] = Path(args.resume) if args.resume else None
    if resume_path is None and args.auto_resume:
        latest = out_dir / "ckpt_latest.pt"
        if latest.exists():
            resume_path = latest

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        tokens_seen = int(ckpt["tokens_seen"])
        print(f"[CinLM] Resumed from {resume_path} at step {start_step} ({tokens_seen:,} tokens)")

    enable_compile = args.compile and device.type == "cuda" and cc_major >= 8
    run_model = torch.compile(model) if enable_compile else model

    data_iter: Optional[object] = None
    if not args.synthetic:
        assert args.tokenizer is not None, "Provide --tokenizer path or pass --synthetic"
        # Offset shuffle seed on resume so streamed documents don't repeat from scratch
        ds = PackedTokenStreamDataset(args.tokenizer, seq_len=seq_len, seed=args.seed + start_step)
        loader = DataLoader(ds, batch_size=args.batch_size)
        data_iter = iter(loader)

    run_model.train()
    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    t0 = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum):
            if args.synthetic:
                x, y = synthetic_batch(cfg.vocab_size, args.batch_size, seq_len, device)
            else:
                x, y = next(data_iter)  # type: ignore[arg-type]
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                out = run_model(x, labels=y)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            loss_accum += float(loss.detach().item())

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        scaler.step(optimizer)
        scaler.update()
        model.update_moe_router_biases()
        tokens_seen += tokens_per_step

        if (step + 1) % args.log_every == 0 or step == start_step:
            dt = time.perf_counter() - t0
            tok_s = (tokens_per_step * (args.log_every if step > start_step else 1)) / max(dt, 1e-6)
            rec = {
                "step": step + 1,
                "loss": round(loss_accum, 5),
                "lr": round(lr, 7),
                "grad_norm": round(grad_norm, 4),
                "tokens_seen": tokens_seen,
                "tok_per_sec": round(tok_s, 1),
            }
            print(json.dumps(rec))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            t0 = time.perf_counter()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            payload = {
                "config": asdict(cfg),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "tokens_seen": tokens_seen,
            }
            # Atomic write to ckpt_latest.pt so a mid-save Colab disconnect never corrupts it
            latest_tmp = out_dir / "ckpt_latest.pt.tmp"
            latest_path = out_dir / "ckpt_latest.pt"
            torch.save(payload, latest_tmp)
            os.replace(latest_tmp, latest_path)

            # Also keep a rotated numbered snapshot (e.g., last 2) without filling Google Drive
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


if __name__ == "__main__":
    main()
