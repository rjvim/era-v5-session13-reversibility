# CPU pilot — reversibility, measured end to end

> **Scope.** 1 CPU core, fp32, tiny-shakespeare (GPT-2 BPE remapped to the 11,706 ids that actually occur), 12 layers / 192 d_model / seq 256, 118,784 tokens per arm. This is a pilot, not the assignment's 20M-param / 50M-token GPU study — it exists to verify the pipeline and establish the relationships before spending GPU hours. Peak-memory columns are GPU-only and are therefore absent here; activation memory is measured device-independently in `results/saved_bytes_scan.json` instead.

| Arm | Batch | Final train loss | Val loss | tok/s | Max recon err |
|---|---|---|---|---|---|
| Baseline | 8 | 6.5841 | 6.4998 | 1,599 | n/a |
| Reversible - euler | 8 | 6.6070 | 6.5105 | 758 | 7.63e+00 |
| Reversible - midpoint | 8 | 6.5673 | 6.4792 | 1,295 | 5.87e-05 |
| Reversible - midpoint @ 2x batch | 16 | 7.3926 | 7.2568 | 1,291 | 8.81e-06 |

## What this pilot already establishes

1. **Midpoint costs 23% throughput** (1,599 -> 1,295 tok/s) for a val-loss difference of -0.0206.
2. **Euler costs 111% throughput** (758 tok/s) for val loss +0.0107 vs baseline — the extra cost is its fixed-point inversion, not extra modelling capacity.
3. **Reconstruction error during real training:** midpoint 5.87e-05 vs euler 7.63e+00 (129860x worse).
4. **Doubling the batch** (the memory headroom reversibility buys) moved val loss from 6.4792 to 7.2568 at the same token budget — half as many optimiser steps.
