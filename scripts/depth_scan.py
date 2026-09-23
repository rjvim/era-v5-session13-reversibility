"""Peak memory vs depth, at fixed batch and sequence length.

This is the measurement that isolates what reversibility actually buys:
baseline activation memory grows with n_layer, reversible memory does not.
Anything OOM is recorded as null rather than dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model import GPT, GPTConfig  # noqa: E402


def peak_for(mode, n_layer, B, T, vocab, n_embd, n_head, h, device, dtype):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        cfg = GPTConfig(vocab_size=vocab, block_size=T, n_layer=n_layer,
                        n_head=n_head, n_embd=n_embd, mode=mode, h=h)
        m = GPT(cfg).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (
            dtype == "bf16" and device.startswith("cuda")) else torch.autocast("cpu", enabled=False)
        for _ in range(2):
            x = torch.randint(0, vocab, (B, T), device=device)
            with ctx:
                _, loss = m(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        pk = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.startswith("cuda") else None
        del m, opt
        return pk
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results_dir", default="results")
    a = p.parse_args()

    rows = []
    for L in a.layers:
        row = {"n_layer": L}
        for mode in ("baseline", "midpoint"):
            row[mode] = peak_for(mode, L, a.batch_size, a.seq_len, a.vocab_size,
                                 a.n_embd, a.n_head, a.h, a.device, a.dtype)
        print(row, flush=True)
        rows.append(row)

    out = {"batch_size": a.batch_size, "seq_len": a.seq_len, "n_embd": a.n_embd,
           "dtype": a.dtype, "rows": rows,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "depth_scaling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote depth_scaling.json")


if __name__ == "__main__":
    main()
