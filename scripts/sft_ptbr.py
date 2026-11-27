#!/usr/bin/env python3
"""PT-BR Conversational Supervised Fine-Tuning (SFT) for CinLM.

Fine-tunes a pretrained CinLM checkpoint on open PT-BR conversational datasets
(Polygl0t/gigaverbo-v2-sft + dominguesm/Canarim-Instruct-PTBR-Dataset) with strict
assistant-only cross-entropy masking (`-100` on system & user tokens).
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

from cinlm.config import CinLMConfig
from cinlm.data import PackedSFTStreamDataset
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
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim < 2 or "tables" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CinLM PT-BR Conversational SFT")
    parser.add_argument("--base-checkpoint", required=True, help="Pretrained CinLM .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to CinLM tokenizer.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--min-lr", type=float, default=4e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--out-dir", default="runs/cinlm_sft_ptbr")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        cc_major, _ = torch.cuda.get_device_capability(0)
        if cc_major >= 8 and torch.cuda.is_bf16_supported():
            autocast_dtype, use_amp, use_scaler = torch.bfloat16, True, False
        else:
            autocast_dtype, use_amp, use_scaler = torch.float16, True, True
    else:
        autocast_dtype, use_amp, use_scaler = torch.float32, False, False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sft_log.jsonl"

    # Load from SFT auto-resume checkpoint if present, otherwise initialize weights from base checkpoint
    sft_latest = out_dir / "sft_latest.pt"
    resuming_sft = args.auto_resume and sft_latest.exists()
    load_path = sft_latest if resuming_sft else Path(args.base_checkpoint)

    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    cfg = CinLMConfig(**cfg_dict)

    model = CinLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.gradient_checkpointing = args.grad_checkpoint

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    tokens_seen = 0
    if resuming_sft:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        tokens_seen = int(ckpt["tokens_seen"])
        print(f"[CinLM SFT] Resumed from {sft_latest} at step {start_step} ({tokens_seen:,} tokens)")
    else:
        print(
            f"[CinLM SFT] Initialized from pretrained {load_path} "
            f"(pretrain step={int(ckpt.get('step', 0)) + 1}, pretrain tokens={int(ckpt.get('tokens_seen', 0)):,})"
        )

    ds = PackedSFTStreamDataset(args.tokenizer, seq_len=args.seq_len, seed=args.seed + start_step)
    n_workers = min(4, max(2, (os.cpu_count() or 2) // 2)) if device.type == "cuda" else 0
    loader_kwargs = (
        {"num_workers": n_workers, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True}
        if n_workers > 0
        else {}
    )
    loader = DataLoader(ds, batch_size=args.batch_size, **loader_kwargs)
    data_iter = iter(loader)

    model.train()
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    t0 = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                out = model(x, labels=y)
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
                "sft_loss": round(loss_accum, 5),
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
                "stage": "sft_ptbr",
            }
            tmp_path = out_dir / "sft_latest.pt.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, sft_latest)

            ckpt_path = out_dir / f"sft_step_{step + 1}.pt"
            torch.save(payload, ckpt_path)
            if args.keep_checkpoints > 0:
                numbered = sorted(
                    out_dir.glob("sft_step_*.pt"),
                    key=lambda p: int(p.stem.split("_")[-1]),
                )
                for old in numbered[: -args.keep_checkpoints]:
                    old.unlink(missing_ok=True)
            print(f"[CinLM SFT] Saved checkpoint: {ckpt_path}")
            t0 = time.perf_counter()


if __name__ == "__main__":
    main()
