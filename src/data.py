"""Token stream for the 50M-token budget.

Writes a flat uint16 memmap (GPT-2 vocab fits in uint16) so every run reads
exactly the same tokens in exactly the same order -- the arms differ only in
the integrator, never in the data.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

DEFAULT_TOKENS = 50_000_000
VAL_TOKENS = 1_000_000


def prepare(out_dir: str, n_tokens: int = DEFAULT_TOKENS,
            dataset: str = "HuggingFaceFW/fineweb-edu", name: str = "sample-10BT"):
    import tiktoken
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    total = n_tokens + VAL_TOKENS
    arr = np.empty(total, dtype=np.uint16)
    i = 0
    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    for doc in ds:
        ids = [eot] + enc.encode_ordinary(doc["text"])
        take = min(len(ids), total - i)
        arr[i:i + take] = np.array(ids[:take], dtype=np.uint16)
        i += take
        if i >= total:
            break
    if i < total:
        raise RuntimeError(f"stream exhausted at {i}/{total} tokens")

    arr[:n_tokens].tofile(os.path.join(out_dir, "train.bin"))
    arr[n_tokens:].tofile(os.path.join(out_dir, "val.bin"))
    meta = {"n_train": int(n_tokens), "n_val": int(VAL_TOKENS),
            "dataset": dataset, "config": name, "encoding": "gpt2"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    return meta


class TokenLoader:
    """Deterministic sequential loader. Identical token order for every arm."""

    def __init__(self, path: str, batch_size: int, seq_len: int, device: str = "cuda"):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T, self.device = batch_size, seq_len, device
        self.pos = 0
        self.stride = batch_size * seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.stride

    def next_batch(self):
        import torch
        if self.pos + self.stride + 1 > len(self.data):
            self.pos = 0
        buf = self.data[self.pos:self.pos + self.stride + 1].astype(np.int64)
        self.pos += self.stride
        x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        y = torch.from_numpy(buf[1:]).view(self.B, self.T)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def reset(self):
        self.pos = 0


def make_synthetic(out_dir: str, n_tokens: int = 2_000_000, vocab: int = 50257, seed: int = 0):
    """Offline fallback used by CI / smoke tests (no network)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rng.integers(0, vocab, size=n_tokens, dtype=np.uint16).tofile(
        os.path.join(out_dir, "train.bin"))
    rng.integers(0, vocab, size=n_tokens // 10, dtype=np.uint16).tofile(
        os.path.join(out_dir, "val.bin"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        make_synthetic(a.out_dir, a.tokens)
    else:
        print(prepare(a.out_dir, a.tokens))
