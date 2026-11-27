from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

SPECIAL_TOKENS: List[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def train_ptbr_bpe_tokenizer(
    text_iterator: Iterable[str],
    output_path: str | Path,
    vocab_size: int = 32768,
    min_frequency: int = 2,
) -> Path:
    """Train a byte-level BPE tokenizer tailored for PT-BR (plus technical EN/code).

    Saves a single `tokenizer.json` file loadable directly by Hugging Face `tokenizers`.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return out_path


@dataclass
class StreamSource:
    dataset_name: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    weight: float = 1.0


DEFAULT_PTBR_PRETRAIN_SOURCES: Tuple[StreamSource, ...] = (
    StreamSource(
        dataset_name="Polygl0t/gigaverbo-v2",
        subset=None,
        split="train",
        text_field="text",
        weight=0.70,
    ),
    StreamSource(
        dataset_name="Polygl0t/gigaverbo-v2-synth",
        subset=None,
        split="train",
        text_field="text",
        weight=0.15,
    ),
    StreamSource(
        dataset_name="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        split="train",
        text_field="text",
        weight=0.15,
    ),
)


class PackedTokenStreamDataset(IterableDataset):
    """Streaming token-packing dataset for Colab G4 without requiring terabytes of disk.

    Packs consecutive documents separated by `<|eos|>` into fixed `seq_len + 1` chunks
    and returns `(input_ids, labels)` tensors of length `seq_len`.
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        sources: Tuple[StreamSource, ...] = DEFAULT_PTBR_PRETRAIN_SOURCES,
        seq_len: int = 2048,
        seed: int = 42,
        shuffle_buffer: int = 1000,
    ) -> None:
        super().__init__()
        self.tokenizer_path = str(tokenizer_path)
        self.sources = sources
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _open_streams(self) -> List[Tuple[StreamSource, Iterator[Dict]]]:
        from datasets import load_dataset

        opened = []
        for idx, src in enumerate(self.sources):
            ds = load_dataset(
                src.dataset_name,
                src.subset,
                split=src.split,
                streaming=True,
            )
            if self.shuffle_buffer > 0:
                ds = ds.shuffle(seed=self.seed + idx, buffer_size=self.shuffle_buffer)
            opened.append((src, iter(ds)))
        return opened

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        import random
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(self.tokenizer_path)
        eos_id = tok.token_to_id("<|eos|>")
        assert eos_id is not None, "Tokenizer missing <|eos|> token"

        rng = random.Random(self.seed)
        streams = self._open_streams()
        weights = [s.weight for s, _ in streams]
        buf: List[int] = []

        while streams:
            chosen_idx = rng.choices(range(len(streams)), weights=weights, k=1)[0]
            src, it = streams[chosen_idx]
            try:
                sample = next(it)
            except StopIteration:
                streams.pop(chosen_idx)
                weights.pop(chosen_idx)
                continue

            text = sample.get(src.text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue

            ids = tok.encode(text).ids
            ids.append(eos_id)
            buf.extend(ids)

            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield t, t.clone()


def format_chat_example(
    messages: List[Dict[str, str]],
    system_prompt: str = "Você é o CinLM, um assistente útil, claro e direto em português do Brasil.",
) -> str:
    """Format multi-turn PT-BR SFT conversation using CinLM chat tags."""
    parts = [f"<|im_start|><|system|>\n{system_prompt}<|im_end|>\n"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        role_tag = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"<|im_start|>{role_tag}\n{content}<|im_end|>\n")
    return "".join(parts)
