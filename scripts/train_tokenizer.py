#!/usr/bin/env python3
"""Train a compact 32k PT-BR (+ optional EN) BPE tokenizer from Hugging Face streaming datasets."""

from __future__ import annotations

import argparse
from itertools import islice

from datasets import load_dataset

from cinlm.data import train_ptbr_bpe_tokenizer


def iter_texts(dataset_name: str, subset: str | None, text_field: str, max_docs: int):
    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    for row in islice(ds, max_docs):
        text = row.get(text_field, "")
        if isinstance(text, str) and text.strip():
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CinLM PT-BR BPE tokenizer")
    parser.add_argument("--dataset", default="Polygl0t/gigaverbo-v2")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-docs", type=int, default=200_000)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--output", default="artifacts/cinlm_bpe_32k.json")
    args = parser.parse_args()

    print(f"Streaming up to {args.max_docs:,} docs from {args.dataset} ...")
    out_path = train_ptbr_bpe_tokenizer(
        iter_texts(args.dataset, args.subset, args.text_field, args.max_docs),
        output_path=args.output,
        vocab_size=args.vocab_size,
    )
    print(f"Saved tokenizer to {out_path}")


if __name__ == "__main__":
    main()
