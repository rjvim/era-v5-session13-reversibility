"""Reconstruction error vs weight scale and dtype. Runs on CPU in seconds.

Backs two README claims with measurement rather than assertion:
  - euler's inverse stops converging once h * Lip(f) > 1, midpoint's does not
  - bf16 imposes a precision floor on BOTH variants
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import GPT, GPTConfig  # noqa: E402

CFG = dict(vocab_size=128, block_size=32, n_layer=8, n_head=2, n_embd=64)


def err(mode, h, scale, iters=8, dtype=torch.float32, seed=3):
    cfg = GPTConfig(**CFG, mode=mode, h=h, euler_iters=iters)
    torch.manual_seed(seed)
    m = GPT(cfg).to(dtype)
    with torch.no_grad():
        for p in m.layers.parameters():
            p.mul_(scale)
    idx = torch.randint(0, CFG["vocab_size"], (2, CFG["block_size"]))
    _, loss = m(idx, idx, check_recon=True)
    loss.backward()
    return m.last_recon_err


def main():
    scales = [1, 2, 5, 10, 20, 40]
    out = {
        "config": CFG, "h": 0.5, "euler_iters": 8,
        "fp32_vs_weight_scale": [
            {"scale": s, "euler": err("euler", 0.5, s), "midpoint": err("midpoint", 0.5, s)}
            for s in scales
        ],
        "bf16_vs_weight_scale": [
            {"scale": s,
             "euler": err("euler", 0.5, s, dtype=torch.bfloat16),
             "midpoint": err("midpoint", 0.5, s, dtype=torch.bfloat16)}
            for s in [1, 5, 20]
        ],
        "fp32_euler_vs_iterations": [
            {"iters": k, "euler": err("euler", 0.5, 5, iters=k)} for k in [1, 2, 4, 8, 16, 32]
        ],
        "torch": torch.__version__,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "precision_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    for row in out["fp32_vs_weight_scale"]:
        print(f"scale {row['scale']:>3}  euler {row['euler']:.3e}  midpoint {row['midpoint']:.3e}")
    print("wrote results/precision_probe.json")


if __name__ == "__main__":
    main()
