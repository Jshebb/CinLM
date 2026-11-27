#!/usr/bin/env python3
"""Stage 3 Post-Training: Reference-Free SimPO Preference Alignment for CinLM (Apache 2.0).

Implements Simple Preference Optimization (Meng et al., NeurIPS 2024, arXiv:2405.14734):
  L_SimPO = -log sigma( (beta/|y_w|) log pi(y_w|x) - (beta/|y_l|) log pi(y_l|x) - gamma )
            + sft_weight * NLL(y_w|x)

Why SimPO for CinLM-1B-A200M:
- Zero reference model in VRAM (runs comfortably on a single A100-80GB).
- Directly eliminates small-model failure modes: repetition loops, rambling, and ungrounded
  hallucinations, while preserving clean PT-BR SFT quality via the NLL anchor term.

Supports two data sources:
1. `--pairs-file pairs.jsonl` with explicit `{"prompt", "chosen", "rejected"}` rows.
2. Default (no `--pairs-file`): 100% Apache 2.0 self-contrastive PT-BR stream from
   `Polygl0t/gigaverbo-v2-sft` (`general`, `retrieval`, `reasoning`), pairing clean ground-truth
   answers (`chosen`) against realistic degenerate negatives (`rejected`: repetition loops,
   truncated rambling, or off-topic/perturbed responses).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cinlm.config import CinLMConfig
from cinlm.data import encode_sft_conversation, normalize_sft_row
from cinlm.model import CinLM
from cinlm.muon import build_cinlm_optimizer


def make_degenerate_negative(chosen_text: str, rng: random.Random, distractor_pool: List[str]) -> str:
    """Synthesize realistic small-LM failure modes (repetition loops, rambling, or off-topic drift)
    to contrast against clean PT-BR assistant answers.
    """
    text = chosen_text.strip()
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 12]
    mode = rng.random()

    # 1. Classic small-LM repetition loop (50%)
    if mode < 0.50 and sentences:
        seed_sent = sentences[0]
        clause = seed_sent.split(",")[0].strip()
        return f"{seed_sent}. {seed_sent}. {clause}, {clause}, {clause}."

    # 2. Word/phrase stutter loop (25%)
    words = text.split()
    if mode < 0.75 and len(words) >= 8:
        prefix = " ".join(words[: min(14, len(words))])
        tail = " ".join(words[max(0, min(14, len(words)) - 5) : min(14, len(words))])
        return f"{prefix} {tail} {tail} {tail}..."

    # 3. Off-topic / ungrounded drift from another answer in the pool (25%)
    if distractor_pool:
        other = rng.choice(distractor_pool)
        if other.strip() != text:
            return other.strip()

    # Fallback repetition
    return f"{text[:120]} {text[:120]} {text[:80]}"


def compute_simpo_loss(
    model: CinLM,
    chosen_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    *,
    beta: float = 2.0,
    gamma: float = 0.5,
    sft_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute reference-free SimPO loss + optional NLL anchor on chosen completion."""
    def _avg_logprob_and_nll(ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = model(ids)["logits"]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        safe_labels = shift_labels.clamp(min=0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        tok_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp * valid.float()
        lengths = valid.float().sum(dim=-1).clamp(min=1.0)
        avg_lp = tok_lp.sum(dim=-1) / lengths
        nll = -(tok_lp.sum() / valid.float().sum().clamp(min=1.0))
        return avg_lp, nll

    pi_w, nll_w = _avg_logprob_and_nll(chosen_ids, chosen_labels)
    pi_l, _ = _avg_logprob_and_nll(rejected_ids, rejected_labels)

    margin = beta * (pi_w - pi_l) - gamma
    simpo_loss = -F.logsigmoid(margin).mean()
    total_loss = simpo_loss + sft_weight * nll_w

    metrics = {
        "simpo_loss": float(simpo_loss.detach().item()),
        "nll_chosen": float(nll_w.detach().item()),
        "reward_margin": float((pi_w - pi_l).mean().detach().item()),
        "pref_acc": float((pi_w > pi_l).float().mean().detach().item()),
    }
    return total_loss, metrics


def _pad_pair_batch(
    items: List[Tuple[List[int], List[int], List[int], List[int]]],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_w = max(len(c_ids) for c_ids, _, _, _ in items)
    max_l = max(len(r_ids) for _, _, r_ids, _ in items)

    c_ids_b, c_lbl_b, r_ids_b, r_lbl_b = [], [], [], []
    for c_ids, c_lbl, r_ids, r_lbl in items:
        c_ids_b.append(c_ids + [pad_id] * (max_w - len(c_ids)))
        c_lbl_b.append(c_lbl + [-100] * (max_w - len(c_lbl)))
        r_ids_b.append(r_ids + [pad_id] * (max_l - len(r_ids)))
        r_lbl_b.append(r_lbl + [-100] * (max_l - len(r_lbl)))

    return (
        torch.tensor(c_ids_b, dtype=torch.long, device=device),
        torch.tensor(c_lbl_b, dtype=torch.long, device=device),
        torch.tensor(r_ids_b, dtype=torch.long, device=device),
        torch.tensor(r_lbl_b, dtype=torch.long, device=device),
    )


def iter_preference_pairs(
    tok: Any,
    max_seq_len: int,
    seed: int,
    pairs_file: Optional[str] = None,
) -> Iterator[Tuple[List[int], List[int], List[int], List[int]]]:
    encode_fn = lambda s: tok.encode(s).ids
    rng = random.Random(seed)

    if pairs_file:
        rows = [json.loads(line) for line in Path(pairs_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        while True:
            rng.shuffle(rows)
            for row in rows:
                prompt = str(row.get("prompt", "")).strip()
                chosen = str(row.get("chosen", "")).strip()
                rejected = str(row.get("rejected", "")).strip()
                if not prompt or not chosen or not rejected:
                    continue
                c_ids, c_lbl = encode_sft_conversation(
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
                    encode_fn=encode_fn,
                    max_seq_len=max_seq_len,
                )
                r_ids, r_lbl = encode_sft_conversation(
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
                    encode_fn=encode_fn,
                    max_seq_len=max_seq_len,
                )
                if any(t != -100 for t in c_lbl[1:]) and any(t != -100 for t in r_lbl[1:]):
                    yield c_ids, c_lbl, r_ids, r_lbl
        return

    from datasets import load_dataset

    sources = [
        ("Polygl0t/gigaverbo-v2-sft", "general", 0.50),
        ("Polygl0t/gigaverbo-v2-sft", "retrieval", 0.30),
        ("Polygl0t/gigaverbo-v2-sft", "reasoning", 0.20),
    ]
    epochs = [0] * len(sources)
    streams = []
    weights = [w for _, _, w in sources]
    for i, (name, cfg_name, _) in enumerate(sources):
        ds = load_dataset(name, cfg_name, split="train", streaming=True).shuffle(seed=seed + i, buffer_size=1000)
        streams.append(iter(ds))

    distractor_pool: List[str] = []

    while True:
        idx = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        try:
            row = next(streams[idx])
        except StopIteration:
            epochs[idx] += 1
            name, cfg_name, _ = sources[idx]
            ds = load_dataset(name, cfg_name, split="train", streaming=True).shuffle(
                seed=seed + idx + epochs[idx] * 997, buffer_size=1000
            )
            streams[idx] = iter(ds)
            continue

        msgs = normalize_sft_row(row)
        if not msgs or len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            continue
        chosen_text = msgs[-1]["content"].strip()
        if len(chosen_text) < 25:
            continue

        rejected_text = make_degenerate_negative(chosen_text, rng, distractor_pool)
        if len(distractor_pool) < 128:
            distractor_pool.append(chosen_text)
        else:
            distractor_pool[rng.randrange(128)] = chosen_text

        rejected_msgs = msgs[:-1] + [{"role": "assistant", "content": rejected_text}]
        c_ids, c_lbl = encode_sft_conversation(msgs, encode_fn=encode_fn, max_seq_len=max_seq_len)
        r_ids, r_lbl = encode_sft_conversation(rejected_msgs, encode_fn=encode_fn, max_seq_len=max_seq_len)
        if any(t != -100 for t in c_lbl[1:]) and any(t != -100 for t in r_lbl[1:]):
            yield c_ids, c_lbl, r_ids, r_lbl


def main() -> None:
    parser = argparse.ArgumentParser(description="CinLM Stage 3 Post-Training: SimPO Preference Alignment")
    parser.add_argument("--sft-checkpoint", required=True, help="Path to SFT CinLM .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to cinlm_bpe_32k.json")
    parser.add_argument("--pairs-file", default=None, help="Optional JSONL {prompt, chosen, rejected}")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--sft-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=150)
    parser.add_argument("--out-dir", default="runs/cinlm_simpo_ptbr")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "simpo_log.jsonl"

    ckpt = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    cfg = CinLMConfig(**cfg_dict)
    model = CinLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    from tokenizers import Tokenizer

    optimizer, _ = build_cinlm_optimizer(model, optimizer_type="adamw", lr=args.lr, weight_decay=0.01)
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<|pad|>") or 0
    pair_iter = iter_preference_pairs(tok, max_seq_len=args.seq_len, seed=args.seed, pairs_file=args.pairs_file)

    print(f"[CinLM SimPO] Loaded {args.sft_checkpoint} | steps={args.max_steps} beta={args.beta} gamma={args.gamma}")
    t0 = time.perf_counter()

    for step in range(args.max_steps):
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
            lr = args.lr * 0.1 + 0.5 * (args.lr * 0.9) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "simpo_loss": 0.0, "nll_chosen": 0.0, "reward_margin": 0.0, "pref_acc": 0.0}

        for _ in range(args.grad_accum):
            batch_items = [next(pair_iter) for _ in range(args.batch_size)]
            c_ids, c_lbl, r_ids, r_lbl = _pad_pair_batch(batch_items, pad_id, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss, m = compute_simpo_loss(
                    model,
                    c_ids,
                    c_lbl,
                    r_ids,
                    r_lbl,
                    beta=args.beta,
                    gamma=args.gamma,
                    sft_weight=args.sft_weight,
                )
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            agg["loss"] += float(scaled_loss.detach().item())
            for k in ("simpo_loss", "nll_chosen", "reward_margin", "pref_acc"):
                agg[k] += m[k] / args.grad_accum

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        optimizer.step()
        model.update_moe_router_biases()

        if (step + 1) % args.log_every == 0 or step == 0:
            rec = {
                "step": step + 1,
                "loss": round(agg["loss"], 5),
                "simpo_loss": round(agg["simpo_loss"], 5),
                "nll_chosen": round(agg["nll_chosen"], 5),
                "reward_margin": round(agg["reward_margin"], 4),
                "pref_acc": round(agg["pref_acc"], 4),
                "lr": round(lr, 8),
                "grad_norm": round(grad_norm, 4),
                "elapsed_sec": round(time.perf_counter() - t0, 1),
            }
            print(json.dumps(rec))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            payload = {
                "config": asdict(cfg),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "tokens_seen": int(ckpt.get("tokens_seen", 0)),
                "stage": "simpo_ptbr",
            }
            tmp = out_dir / "simpo_latest.pt.tmp"
            latest = out_dir / "simpo_latest.pt"
            torch.save(payload, tmp)
            os.replace(tmp, latest)
            snap = out_dir / f"simpo_step_{step + 1}.pt"
            torch.save(payload, snap)
            print(f"[CinLM SimPO] Saved checkpoint: {snap}")


if __name__ == "__main__":
    main()
