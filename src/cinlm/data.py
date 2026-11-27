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

    def _open_streams(self, seed_offset: int = 0) -> List[Tuple[StreamSource, Iterator[Dict]]]:
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
                ds = ds.shuffle(seed=self.seed + seed_offset + idx, buffer_size=self.shuffle_buffer)
            opened.append((src, iter(ds)))
        return opened

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        import random
        from tokenizers import Tokenizer

        worker_info = torch.utils.data.get_worker_info()
        worker_offset = worker_info.id * 1009 if worker_info is not None else 0

        tok = Tokenizer.from_file(self.tokenizer_path)
        eos_id = tok.token_to_id("<|eos|>")
        assert eos_id is not None, "Tokenizer missing <|eos|> token"

        rng = random.Random(self.seed + worker_offset)
        streams = self._open_streams(seed_offset=worker_offset)
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


def encode_sft_conversation(
    messages: List[Dict[str, str]],
    encode_fn,
    max_seq_len: int,
    system_prompt: str = "Você é o CinLM, um assistente útil, claro e direto em português do Brasil.",
) -> Tuple[List[int], List[int]]:
    """Tokenize a multi-turn conversation and mask non-assistant tokens with -100.

    Only assistant content tokens and their closing `<|im_end|>` contribute to SFT loss.
    `encode_fn` maps a string -> List[int].
    """
    input_ids: List[int] = []
    labels: List[int] = []

    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system and system_prompt:
        sys_ids = encode_fn(f"<|im_start|><|system|>\n{system_prompt}<|im_end|>\n")
        input_ids.extend(sys_ids)
        labels.extend([-100] * len(sys_ids))

    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            seg = encode_fn(f"<|im_start|><|system|>\n{content}<|im_end|>\n")
            input_ids.extend(seg)
            labels.extend([-100] * len(seg))
        elif role in ("user", "human"):
            seg = encode_fn(f"<|im_start|><|user|>\n{content}<|im_end|>\n")
            input_ids.extend(seg)
            labels.extend([-100] * len(seg))
        elif role in ("assistant", "gpt", "model"):
            header = encode_fn("<|im_start|><|assistant|>\n")
            body = encode_fn(f"{content}<|im_end|>\n")
            input_ids.extend(header)
            labels.extend([-100] * len(header))
            input_ids.extend(body)
            labels.extend(body)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    return input_ids, labels


def normalize_sft_row(row: Dict) -> Optional[List[Dict[str, str]]]:
    """Normalize rows from GigaVerbo-v2-SFT (`messages`) or Canarim (`instruction`/`context`/`output`)."""
    if "messages" in row and isinstance(row["messages"], list):
        msgs = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in row["messages"]
            if isinstance(m, dict) and m.get("content")
        ]
        if any(m["role"] == "assistant" for m in msgs):
            return msgs
        return None

    if "instruction" in row and "output" in row:
        instr = str(row.get("instruction") or "").strip()
        ctx = str(row.get("context") or row.get("input") or "").strip()
        out = str(row.get("output") or "").strip()
        if not instr or not out:
            return None
        user_msg = f"{instr}\n\nContexto:\n{ctx}" if ctx else instr
        return [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": out},
        ]

    return None


class PackedSFTStreamDataset(IterableDataset):
    """Streaming PT-BR SFT dataset with strict assistant-only loss masking.

    Defaults to 100% Apache 2.0 (`Polygl0t/gigaverbo-v2-sft`, ~4.1M PT-BR conversations)
    so the resulting CinLM-Chat weights can be published cleanly under Apache 2.0.
    Pass `include_canarim=True` to mix 30% `dominguesm/Canarim-Instruct-PTBR-Dataset` (CC-BY-NC-4.0).
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        seq_len: int = 2048,
        seed: int = 42,
        shuffle_buffer: int = 1000,
        include_canarim: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer_path = str(tokenizer_path)
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        if include_canarim:
            self.sources = (
                ("Polygl0t/gigaverbo-v2-sft", None, "train", 0.70),
                ("dominguesm/Canarim-Instruct-PTBR-Dataset", None, "train", 0.30),
            )
        else:
            self.sources = (("Polygl0t/gigaverbo-v2-sft", None, "train", 1.0),)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        import random
        from datasets import load_dataset
        from tokenizers import Tokenizer

        worker_info = torch.utils.data.get_worker_info()
        worker_offset = worker_info.id * 1009 if worker_info is not None else 0

        tok = Tokenizer.from_file(self.tokenizer_path)
        encode_fn = lambda s: tok.encode(s).ids

        rng = random.Random(self.seed + worker_offset)
        streams = []
        weights = []
        for idx, (name, subset, split, w) in enumerate(self.sources):
            ds = load_dataset(name, subset, split=split, streaming=True)
            if self.shuffle_buffer > 0:
                ds = ds.shuffle(seed=self.seed + worker_offset + idx, buffer_size=self.shuffle_buffer)
            streams.append(iter(ds))
            weights.append(w)

        id_buf: List[int] = []
        lbl_buf: List[int] = []

        while streams:
            chosen = rng.choices(range(len(streams)), weights=weights, k=1)[0]
            try:
                row = next(streams[chosen])
            except StopIteration:
                streams.pop(chosen)
                weights.pop(chosen)
                continue

            msgs = normalize_sft_row(row)
            if not msgs:
                continue

            ids, lbls = encode_sft_conversation(msgs, encode_fn=encode_fn, max_seq_len=self.seq_len)
            if not any(l != -100 for l in lbls):
                continue

            id_buf.extend(ids)
            lbl_buf.extend(lbls)

            while len(id_buf) >= self.seq_len:
                x = id_buf[: self.seq_len]
                y = lbl_buf[: self.seq_len]
                id_buf = id_buf[self.seq_len :]
                lbl_buf = lbl_buf[self.seq_len :]
                # Skip pathological chunks with zero supervised tokens
                if any(t != -100 for t in y[1:]):
                    yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
