#!/usr/bin/env python3
"""Run a Frontier-LLM interpretability audit on a CinLM checkpoint and export JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from cinlm.config import CinLMConfig, get_preset_config
from cinlm.interpretability import run_frontier_audit
from cinlm.model import CinLM, InterventionSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Frontier-LLM audit JSON for a PT-BR prompt")
    parser.add_argument("--prompt", required=True, help="Input text in PT-BR")
    parser.add_argument("--tokenizer", required=True, help="Path to CinLM tokenizer.json")
    parser.add_argument("--checkpoint", default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--preset", default="smoke", help="Fallback preset if no checkpoint is given")
    parser.add_argument("--ablate-engram", action="store_true", help="Clamp all Engram gates to 0.0")
    parser.add_argument("--out", default=None, help="Optional path to save audit JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]
        # Restore tuples deserialized as lists from JSON/dict
        for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
            if k in cfg_dict and isinstance(cfg_dict[k], list):
                cfg_dict[k] = tuple(cfg_dict[k])
        cfg = CinLMConfig(**cfg_dict)
        model = CinLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        cfg = get_preset_config(args.preset)  # type: ignore[arg-type]
        model = CinLM(cfg).to(device)

    ids = tok.encode(args.prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    intervention = None
    if args.ablate_engram and cfg.use_engram:
        intervention = InterventionSpec(
            force_engram_gate={l_idx: 0.0 for l_idx in cfg.engram_layers}
        )

    report = run_frontier_audit(
        model,
        input_ids,
        decode_token=lambda tid: tok.decode([tid]),
        target_pos=-1,
        intervention=intervention,
    )
    payload = report.to_json(indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"Saved audit JSON to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
