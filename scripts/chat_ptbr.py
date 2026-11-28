#!/usr/bin/env python3
"""Interactive PT-BR multi-turn chat CLI for SFT-tuned CinLM checkpoints."""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from tokenizers import Tokenizer

from cinlm.config import CinLMConfig
from cinlm.data import format_chat_example
from cinlm.model import CinLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat interactively with CinLM in PT-BR")
    parser.add_argument("--checkpoint", required=True, help="Path to SFT .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to CinLM tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--mode", default="direct", choices=["direct", "think", "auto"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    im_end_id = tok.token_to_id("<|im_end|>")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    model = CinLM(CinLMConfig(**cfg_dict)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    history: List[Dict[str, str]] = []
    print("CinLM Chat PT-BR pronto! Digite 'sair' para encerrar ou 'limpar' para reiniciar o contexto.\n")

    while True:
        user_text = input("Você: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("sair", "exit", "quit"):
            break
        if user_text.lower() == "limpar":
            history.clear()
            print("[Histórico limpo]\n")
            continue

        history.append({"role": "user", "content": user_text})
        prompt = format_chat_example(history, mode=args.mode) + "<|im_start|><|assistant|>\n"
        prompt_ids = tok.encode(prompt).ids[-(model.config.max_seq_len - args.max_new_tokens) :]
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out = model.generate(
            inp,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=im_end_id,
        )
        new_ids = out[0, len(prompt_ids) :].tolist()
        if im_end_id in new_ids:
            new_ids = new_ids[: new_ids.index(im_end_id)]
        reply = tok.decode(new_ids).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"CinLM: {reply}\n")


if __name__ == "__main__":
    main()
