"""Activation bytes actually handed to autograd, vs depth. Device-independent.

CUDA peak memory mixes weights, optimiser states, fragmentation and caching.
This counts only what the backward pass is asked to keep, by hooking autograd's
saved-tensor path -- so the O(1)-in-depth claim can be checked on any machine.
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import GPT, GPTConfig  # noqa: E402


def saved_bytes(mode, n_layer, B=2, T=128, vocab=512, n_embd=128, n_head=4):
    cfg = GPTConfig(vocab_size=vocab, block_size=T, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, mode=mode, h=0.5)
    m = GPT(cfg)
    idx = torch.randint(0, vocab, (B, T))
    tot = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (tot.append(t.numel() * t.element_size()), t)[1], lambda t: t):
        _, loss = m(idx, idx)
    return sum(tot)


if __name__ == "__main__":
    layers = [2, 4, 8, 16, 32, 64]
    rows = [{"n_layer": L,
             "baseline_bytes": saved_bytes("baseline", L),
             "midpoint_bytes": saved_bytes("midpoint", L),
             "euler_bytes": saved_bytes("euler", L)} for L in layers]
    for r in rows:
        print(f"L={r['n_layer']:>3}  baseline {r['baseline_bytes']/1e6:8.2f} MB   "
              f"midpoint {r['midpoint_bytes']/1e6:8.2f} MB")
    out = {"batch": 2, "seq_len": 128, "n_embd": 128, "rows": rows}
    with open(os.path.join(ROOT, "results", "saved_bytes_scan.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/saved_bytes_scan.json")
