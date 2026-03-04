import hashlib
import os
import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    block_size: int = 256
    train_chars: int = 2_000_000
    val_chars: int = 200_000
    data_seed: int = 42
    local_fallback_path: str = "data/local_corpus.txt"


class ByteTokenizer:
    vocab_size = 257
    eos_token = 256

    def encode(self, text: str) -> list[int]:
        data = list(text.encode("utf-8", errors="ignore"))
        data.append(self.eos_token)
        return data


class SequenceDataset(Dataset):
    def __init__(self, tokens: list[int], block_size: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


def _load_text(cfg: DataConfig) -> str:
    try:
        from datasets import load_dataset

        ds = load_dataset("openwebtext", split="train", streaming=True)
        random.seed(cfg.data_seed)
        chars = []
        target = cfg.train_chars + cfg.val_chars
        for row in ds:
            chars.append(row.get("text", ""))
            if sum(len(c) for c in chars) >= target:
                break
        return "\n".join(chars)
    except Exception:
        if not os.path.exists(cfg.local_fallback_path):
            raise FileNotFoundError(
                f"OpenWebText unavailable and fallback file missing at {cfg.local_fallback_path}."
            )
        with open(cfg.local_fallback_path, "r", encoding="utf-8") as f:
            return f.read()


def build_datasets(cfg: DataConfig) -> Tuple[SequenceDataset, SequenceDataset, dict]:
    text = _load_text(cfg)
    text = text[: cfg.train_chars + cfg.val_chars]
    tok = ByteTokenizer()
    ids = tok.encode(text)

    split = int(len(ids) * (cfg.train_chars / (cfg.train_chars + cfg.val_chars)))
    train_ids = ids[:split]
    val_ids = ids[split:]

    split_hash = hashlib.md5((str(cfg) + str(len(ids))).encode()).hexdigest()
    train_ds = SequenceDataset(train_ids, cfg.block_size)
    val_ds = SequenceDataset(val_ids, cfg.block_size)
    meta = {
        "vocab_size": tok.vocab_size,
        "split_hash": split_hash,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    return train_ds, val_ds, meta
