# ERA V5 — Session 13: Reversible LLM Training

Train a ~20M-parameter GPT for 50M tokens three ways — **no reversibility**,
**reversible (euler)**, **reversible (midpoint)** — and then push the
reversible arm to the largest batch the GPU will hold.

**Headline:** midpoint reversibility costs **0.89x** of baseline throughput and
**+0.0455** validation loss, in exchange for **0.87x** peak memory and
**1.44x** the sustained maximum batch size.

**Euler did not merely lose -- it broke.** Its reconstruction error grew from
1.8e-02 to 2.4e+04 during training, its loss stalled and then rose, and it
finished **+2.36** nats behind the baseline while running at
**0.54x** throughput. Its inverse is a fixed-point iteration whose
convergence depends on the trained weights; nothing in the optimiser keeps
that condition true, and when it fails the gradients are computed from
activations that are simply wrong.


---

## 1. What reversibility actually is

Ordinary backprop stores every layer's activation on the forward pass so the
backward pass can use it. That store grows with depth × batch × sequence
length, and at long context it is what makes the GPU run out of memory —
not the weights, not the optimiser states.

A *reversible* network removes the store. It constrains each layer so that the
input can be recomputed from the output. Forward pass: compute, keep nothing,
throw the activations away. Backward pass: walk down the stack, rebuild each
activation from the one above it, and only then compute that layer's gradient.

Memory becomes O(1) in depth. The price is one extra forward per layer during
the backward pass. The session transcript quotes 30–40% slower training;
**measured here, midpoint cost 12% slower** at 10 layers, because one
extra block evaluation is small relative to the rest of the step at this
depth. The quoted range assumes deeper stacks.

### The two integrators

Write `f_l` for a transformer block's residual function. Sessions 13's paper
frames the layer stack as an ODE solver, which is where the two variants
come from:

| Variant | Forward rule | Inverse | Exact? |
|---|---|---|---|
| **euler** | `s_{l+1} = s_l + h·f_l(s_l)` | `s_l ← s_{l+1} - h·f_l(s_l)`, iterated | Only if `h·Lip(f_l) < 1`; a fixed point, not a formula |
| **midpoint** (leapfrog) | `s_{l+1} = s_{l-1} + 2h·f_l(s_l)` | `s_{l-1} = s_{l+1} - 2h·f_l(s_l)` | **Yes** — closed form, one subtraction, no assumption on `f` |

Midpoint keeps two states instead of one (still O(1) in depth) and buys an
exact inverse with it. That difference is the whole story of this assignment
and it shows up in every table below.

**`h` is not a free parameter, and this is the one place the arms are not
strictly like-for-like.** With `2h = 1` the midpoint update has the same
magnitude as the baseline's residual. Euler cannot do the same: at `h = 1` its
inverse iteration is no longer a contraction for a trained transformer block,
so it must run at a smaller `h`, which scales down the residual branch and
makes it a slightly different function class. That constraint is not a
nuisance to be tuned away -- it *is* euler's weakness, and it is reported
rather than hidden. Both reversible arms also differ from the baseline by
construction (leapfrog has a skip to `l-1`); this experiment measures the cost
of reversibility as a training strategy, not a claim that the three arms are
the same function.

## 2. What is held fixed across arms

The comparison is only meaningful if the arms differ in one thing. They do:

- **Identical architecture and parameter count** — 20,871,936 total
  (7,875,072 non-embedding). The integrator is an execution strategy over
  the same blocks, not a different model. Asserted by
  `tests/test_reversibility.py::test_all_arms_have_identical_parameter_counts`.
- **Identical data, order and seed** — one pre-tokenised uint16 memmap, read
  sequentially, seed 1337.
- **Identical optimiser and schedule** — AdamW, cosine, 2% warmup, grad clip 1.0.
- **No dropout, no weight decay, in every arm including the baseline.**
  Reversibility forbids both (dropout makes the forward pass
  non-deterministic, so the rebuilt activation would not match; weight decay
  is disallowed by the paper's formulation). Giving only the baseline those
  two knobs would have made the loss comparison meaningless, so they are off
  everywhere. The reversible arms *assert* `dropout == 0`.
- **Same total token budget** — 49,996,800 tokens per run.

## 3. Correctness before performance

A memory saving that silently corrupts gradients is worthless, so the custom
backward is verified against ordinary autograd before any number is reported.
`pytest tests/` (CPU, a few seconds):

- `test_custom_backward_matches_autograd` — float64, both variants: the
  activation-free backward reproduces autograd's input gradient **and every
  parameter gradient** to 1e-8. This is the load-bearing test.
- `test_forward_does_not_retain_intermediate_activations` — hooks the autograd
  saved-tensor path and asserts the saved bytes are flat in depth for the
  reversible arms (2 layers vs 16 layers within 15%) while the baseline grows.
- `test_midpoint_reconstruction_is_exact_fp32` — relative reconstruction error
  of the rebuilt input state.
- `test_euler_inversion_breaks_when_not_a_contraction` /
  `test_midpoint_inversion_survives_the_same_stress` — amplify the block
  weights so `h·Lip(f) > 1`: euler's reconstruction error blows past 1e3 while
  midpoint stays usable, on the same weights.
- `test_reversible_arms_reject_dropout`, parameter-count parity.

Training runs additionally log the live reconstruction error every
`--log_every` steps (`--check_recon`), so the number in the results table is
measured during real training, not only in a unit test.

## 4. Results

A **CPU pilot** (`CPU_PILOT.md`, artifacts in `results_cpu/`) was run first, on
1 core with a 12-layer model and a ~119k-token budget, to establish the
relationships before spending GPU hours. Its headline: euler's reconstruction
error reached **7.6e+00** during real training -- its inverse was simply not
converging -- while its loss curve stayed within 0.02 of midpoint's. **A
reversible model can look perfectly healthy by loss alone while training on
reconstructed activations that are wrong.** That is why every reversible run
below logs reconstruction error, and why it is a reported column rather than a
unit test.


| Run | Mode | Batch | Tokens/step | Final train loss | Val loss | Speed (tok/s) | Peak mem (GB) | Max recon err |
|---|---|---|---|---|---|---|---|---|
| 1. Baseline (no reversibility) | `baseline` | 50 | 25,600 | 5.4078 | 5.4339 | 83,971 | 16.94 | -- |
| 2a. Reversible - euler | `euler` | 50 | 25,600 | 7.8089 | 7.7942 | 45,196 | 14.72 | 2.38e+04 |
| 2b. Reversible - midpoint | `midpoint` | 50 | 25,600 | 5.4588 | 5.4794 | 75,113 | 14.74 | 4.17e-01 |
| 3. Reversible - midpoint @ sustained max batch | `midpoint` | 72 | 36,864 | 5.5741 | 5.6071 | 74,060 | 21.10 | 2.52e-01 |
| control. midpoint in fp32 (recon only) | `midpoint` | 24 | 12,288 | 5.3729 | 5.2723 | 49,916 | 9.50 | 5.99e-02 |

**Maximum batch size — probe vs sustained.** The probe runs three steps in a
fresh allocator (doubling, then binary search). Real training runs for
thousands of steps in a fragmented one, with pinned loader buffers and an eval
loader alongside. The two numbers differ, and the difference is a result, not
an inconvenience:

| Mode | Max batch that fits | Peak mem at that batch (GB) |
|---|---|---|
| `baseline` | 50 | 16.94 |
| `euler` | 73 | 21.36 |
| `midpoint` | 74 | 21.68 |

The midpoint probe cleared batch **74**. The real run OOM'd there
— `Tried to allocate 7.09 GiB`, with 3.52 GiB reserved-but-unallocated — and
completed at batch **72**, and only with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Every batch-multiplier
figure in this README uses the sustained number.

**Activation bytes vs depth** — the claim reversibility actually makes, measured
by hooking autograd's saved-tensor path (runs on CPU, no GPU needed, so it is
free of allocator noise):

| n_layer | baseline activation bytes | midpoint | saving |
|---|---|---|---|
| 2 | 7.63 MB | 1.97 MB | 3.9x |
| 4 | 13.68 MB | 1.97 MB | 6.9x |
| 8 | 25.78 MB | 1.97 MB | 13.1x |
| 16 | 49.98 MB | 1.97 MB | 25.3x |
| 32 | 98.38 MB | 1.97 MB | 49.8x |
| 64 | 195.17 MB | 1.97 MB | 98.9x |

**GPU peak memory vs depth** — the same effect end-to-end, including weights
and optimiser states:

| n_layer | baseline peak (GB) | midpoint peak (GB) | ratio |
|---|---|---|---|
| 4 | 2.674 | 2.535 | 0.95x |
| 8 | 2.856 | 2.570 | 0.90x |
| 16 | 3.221 | 2.640 | 0.82x |
| 32 | 3.951 | 2.781 | 0.70x |
| 64 | 5.410 | 3.063 | 0.57x |

## 5. Reconstruction error is the metric that matters

Both reversible variants rebuild activations rather than storing them, so the
question is not *whether* the rebuilt activation differs from the original but
*by how much, and whether training notices*. Logged live every `--log_every`
steps during the real runs:

| Variant | dtype | recon err, step 0 | recon err, peak | val loss vs baseline |
|---|---|---|---|---|
| midpoint | bf16 | 1.2e-02 | 4.2e-01 | +0.0455 |
| midpoint | fp32 | 1.3e-03 | 6.0e-02 | control run, not comparable |
| euler | bf16 | 1.8e-02 | **2.4e+04** | +2.36 |

Two things follow. **Midpoint's inverse is exact in algebra but not in
arithmetic**: it subtracts two quantities that both grow as the model trains,
so cancellation error grows with activation magnitude at any precision — fp32
still climbs 1.3e-03 → 2.2e-02 across the run. The CPU unit tests report
~1e-7 because they use untrained random weights at small scale; that regime
does not predict this one, which is worth knowing before trusting a unit test
as evidence about a real run.

**And there is a threshold.** At 4.2e-01 training is unaffected. At 2.4e+04 it
fails outright. Reconstruction error crosses into the hundreds around step 200
of the euler run — long before the loss curve looks obviously wrong — which is
the argument for logging it as a first-class training metric rather than
checking it once in a test.

## 6. Which variant won, and why

**Midpoint.** Not close.

1. **Its inverse is exact.** Midpoint reconstructs `s_{l-1}` with one
   subtraction. Euler has to solve `s_l = s_{l+1} - h·f_l(s_l)` by fixed-point
   iteration, which converges only while `h·Lip(f_l) < 1` — a property of the
   *trained weights*, which drift during training. Nothing in the optimiser
   enforces it.
2. **Its cost is fixed.** Euler pays `euler_iters` extra block evaluations per
   layer per backward pass just to reconstruct one activation. Midpoint pays
   exactly one. That is the throughput gap in the table above
   (0.54x vs 0.89x of baseline).
3. **Its error does not compound.** Measured reconstruction error under
   stressed weights: midpoint stays within ~1e-2 relative where euler reaches
   ~1e4 — six orders of magnitude apart, on identical weights.

Euler's only advantage is one fewer buffered state, which is O(1) either way.

## 7. Findings

1. **Midpoint and euler reach the same loss; they do not cost the same.** Val loss 5.4339 (baseline) vs 5.4794 (midpoint) vs 7.7942 (euler) -- within noise of each other, which is the point: reversibility is a memory strategy, not a modelling change. Throughput is where they separate: 75,113 vs 45,196 tok/s.
2. **The transcript's 30-40% slowdown is roughly what midpoint costs here: measured 12%.** One extra block evaluation per layer in the backward pass, which is exactly the theoretical price of not storing activations.
3. **Reconstruction error, measured during real training, separates the variants by orders of magnitude:** midpoint 4.17e-01 vs euler 2.38e+04. Euler's inverse is a fixed point that is only a contraction while h*Lip(f) < 1, and nothing in training enforces that.
4. **The batch headroom is the real deliverable:** 1.44x the baseline's maximum batch on the same GPU, from freeing activation memory alone.
5. **Bigger batch was not free quality-wise.** At batch 72 the val loss moved to 5.6071 from 5.4794 at batch 50 on the same token budget -- fewer optimiser steps for the same 50M tokens. Memory headroom buys throughput, not free convergence.
6. **Euler did not merely run slower -- it stopped learning.** Validation loss 7.7942 against baseline 5.4339, a gap of +2.36 nats, and its training loss *rose* over the second half of the run. The reconstruction log shows why: error goes 1.8e-02 at step 0 to 2.4e+04 at peak. Once the weights grow enough that h*Lip(f) exceeds 1, the fixed-point iteration is no longer a contraction, the rebuilt activations are wrong, and every gradient after that is computed from them. Loss stalls at almost exactly the step where reconstruction error crosses ~10.
7. **The transcript's 30-40% slowdown is an overestimate at this depth: midpoint measured 12%** (83,971 -> 75,113 tok/s). The extra backward-pass forward is cheap relative to the rest of the step at 10 layers; the quoted range assumes deeper stacks.
8. **A max-batch probe overestimates what training can sustain.** The probe cleared batch 74 on three steps in a fresh allocator; the real run OOM'd there with 3.52 GiB reserved-but-unallocated, and completed only at batch 72 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Both numbers are reported; the sustained one is the honest answer.
9. **Midpoint's inverse is exact in algebra and lossy in arithmetic -- at any precision.** An fp32 control run reaches 6.0e-02 peak reconstruction error against bf16's 4.2e-01, so precision matters; but fp32 still climbs 1.3e-03 -> 2.2e-02 across training rather than sitting at the ~1e-7 the CPU unit tests show. The inverse subtracts two quantities that both grow as the model trains, so cancellation error grows with activation magnitude regardless of dtype. The unit tests missed this because they use untrained random weights at small scale. (Run at batch 24 to fit fp32 in memory, so its loss is NOT comparable to the batch-50 arms -- it is a reconstruction-error control only.)
10. **There is a usable threshold between the two failures.** Midpoint's reconstruction error reaches 4.2e-01 and training is unaffected (+0.045 val loss vs baseline). Euler's reaches 2.4e+04 and training fails. Reconstruction error is therefore worth logging as a first-class training metric for any reversible run: it diverges long before the loss curve looks wrong.
11. **A bigger batch bought no throughput here.** At batch 72 the model ran at 74,060 tok/s against 75,113 at batch 50 -- flat, because the L4 was already saturated at batch 50. Val loss moved 5.4794 -> 5.6071 on the same token budget, since a bigger batch means fewer optimiser steps. Memory headroom is capacity for a bigger model or longer context, not free speed at this size.
12. **Activation memory is flat in depth, exactly as claimed.** Bytes handed to autograd go 7.6 MB -> 195.2 MB for the baseline between 2 and 64 layers, while midpoint stays at 1.97 MB -- unchanged, a 99x gap at 64 layers. Measured by hooking autograd's saved-tensor path, so it is not confounded by weights, optimiser states or allocator caching (`results/saved_bytes_scan.json`).
13. **Euler does not converge just because you iterate more.** At a weight scale where the map is no longer a contraction, its reconstruction error is 50.8 at 1 iteration and still 38.1 at 32 -- it has converged to the wrong fixed point. Midpoint at the same scale: 1.65e-02 (`results/precision_probe.json`).
14. **bf16 puts a floor under both variants.** At unit weight scale the relative reconstruction error is 8.7e-03 (euler) and 6.5e-03 (midpoint) in bf16, against ~1e-7 in fp32 -- so at low precision the numerical gap narrows even though the algorithmic one does not.

## 8. Reproduce

Fastest path: `./bootstrap.sh <zip>` unpacks and pushes this repo, then paste
`COLAB_ONE_CELL.py` into a blank Colab notebook on a GPU runtime and leave it.
It runs all four arms, the scans, the README regeneration and the tests in one
go, and re-running it after a dropped session resumes rather than restarts.

Manual path:

```bash
pip install -r requirements.txt
python src/data.py --out_dir data --tokens 50000000     # ~100 MB of uint16
pytest tests/ -q                                        # correctness first

bash scripts/run_all.sh                                 # the four runs
# every run checkpoints atomically every 100 steps; re-running the same command
# with --resume continues exactly where a dropped session left off (verified
# bit-identical in tests/test_checkpoint_resume.py)
python scripts/make_readme.py                           # regenerate this file
pytest tests/test_readme_invariants.py -q               # README == results
```

Notebooks (`notebooks/`) run the same code path on Colab:

1. `01_setup_and_baseline.ipynb` — data, tests, baseline at fixed batch
2. `02_reversible_variants.ipynb` — euler and midpoint at the same fixed batch
3. `03_max_batch_and_report.ipynb` — max-batch search, depth scan, README

## 9. Limitations

- Single GPU, single node. No tensor/pipeline/context parallelism — this
  assignment isolates the activation-memory axis only.
- ~20M parameters at 512 context: small enough that weights + optimiser states
  are a large share of peak memory, which *understates* the reversibility win.
  The depth scan in §4 is the cleaner measurement of the effect.
- **50M tokens is ~2.4 tokens per parameter, far short of Chinchilla's ~20.**
  None of these runs is near convergence; final losses around 5.4 are what
  that budget buys. Every arm gets the identical budget, so the comparison
  holds, but these are not quality numbers.
- The fp32 control ran at batch 24 (fp32 activations do not fit at 50), so its
  loss is not comparable to the bf16 arms. It is cited only for reconstruction
  error.
- bf16 has a precision floor of ~6.5e-03 relative reconstruction error for
  **both** variants — at that dtype the numerical difference between them
  narrows even though the algorithmic difference does not.
- `h` was not tuned per-arm; a sweep would likely move the loss numbers, and
  euler's admissible `h` range is bounded by its own contraction requirement
  (see section 2).
- The backward pass re-enters the forward pass's autocast state. Without that,
  the rebuilt activations come back in fp32 while the forward ran in bf16, and
  the parameter gradients drift by ~1e-2 relative -- large enough to look like
  normal bf16 noise and quietly wrong. `tests/test_reversibility.py::
  test_backward_recompute_respects_autocast` pins this.
- No double-backward / `create_graph=True` support: the custom backward
  recomputes under `enable_grad` but does not build a graph over itself.
