"""Generate the three Colab notebooks from a single source of truth."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks")
REPO = "https://github.com/rjvim/era-v5-session13-reversibility"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip().splitlines(keepends=True)}


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


SETUP = f"""
!nvidia-smi
!git clone {REPO} repo 2>/dev/null || (cd repo && git pull)
%cd repo
!pip -q install -r requirements.txt
import sys; sys.path.insert(0, 'src')
"""

N1 = nb([
    md("""
# Session 13 — 1/3: setup, correctness, baseline

Trains the ~20M model for 50M tokens **without** reversibility, at the largest
batch this GPU can hold. That batch is then reused unchanged by notebook 2 so
the arms are comparable.
"""),
    code(SETUP),
    md("## Correctness first\n\nThe custom activation-free backward is checked against ordinary autograd "
       "in float64 before any timing number is trusted."),
    code("!pytest tests/test_reversibility.py -q"),
    md("## Data — 50M training tokens (+1M val), GPT-2 encoding\n\n"
       "Written once as a uint16 memmap so every arm reads identical tokens in identical order."),
    code("!python src/data.py --out_dir data --tokens 50000000\n"
         "!ls -la data/"),
    md("## Largest batch the baseline can hold\n\n"
       "Doubling + binary search over a *full* train step (fwd, bwd, optimiser), not a forward pass."),
    code("!python src/train.py --mode baseline --find_max_batch --seq_len 512"),
    md("## Run 1 — baseline @ fixed batch"),
    code("import json\n"
         "BATCH_FIX = json.load(open('results/maxbatch_baseline.json'))['max_batch']\n"
         "print('fixed batch =', BATCH_FIX)\n"
         "!python src/train.py --mode baseline --batch_size {BATCH_FIX} \\\n"
         "    --run_name baseline_fixed --seq_len 512 --total_tokens 50000000"),
    code("d = json.load(open('results/baseline_fixed.json'))\n"
         "print(f\"loss {d['final_train_loss']:.4f} | val {d['final_val_loss']:.4f} | \"\n"
         "      f\"{d['tok_per_s']:,.0f} tok/s | peak {d['peak_mem_gb']:.2f} GB\")"),
    md("Carry `BATCH_FIX` into notebook 2 unchanged."),
])

N2 = nb([
    md("""
# Session 13 — 2/3: reversible arms (euler vs midpoint)

Same model, same data, same seed, **same batch size as the baseline**. The only
change is the inter-layer update rule and the fact that activations are not
stored at all.

`--check_recon` logs the live reconstruction error during training: how far the
rebuilt activation drifts from the real one. That is the number that decides
which variant is trustworthy, not just which is fast.
"""),
    code(SETUP),
    code("import json\n"
         "BATCH_FIX = json.load(open('results/maxbatch_baseline.json'))['max_batch']\n"
         "print('fixed batch =', BATCH_FIX)"),
    md("## Run 2a — euler"),
    code("!python src/train.py --mode euler --batch_size {BATCH_FIX} --run_name euler_fixed \\\n"
         "    --h 0.5 --euler_iters 8 --check_recon --seq_len 512 --total_tokens 50000000"),
    md("## Run 2b — midpoint"),
    code("!python src/train.py --mode midpoint --batch_size {BATCH_FIX} --run_name midpoint_fixed \\\n"
         "    --h 0.5 --check_recon --seq_len 512 --total_tokens 50000000"),
    md("## Compare"),
    code("""
import json, matplotlib.pyplot as plt
runs = {k: json.load(open(f'results/{k}.json'))
        for k in ['baseline_fixed', 'euler_fixed', 'midpoint_fixed']}
for k, d in runs.items():
    print(f"{k:16s} loss {d['final_train_loss']:.4f} val {d['final_val_loss']:.4f} "
          f"{d['tok_per_s']:>9,.0f} tok/s  peak {d['peak_mem_gb']:.2f} GB  "
          f"recon {d['max_recon_err'] if d['max_recon_err'] else 0:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, d in runs.items():
    ax[0].plot([r['step'] for r in d['log']], [r['loss'] for r in d['log']], label=k)
    if d['recon_err_log']:
        ax[1].semilogy([r[0] for r in d['recon_err_log']],
                       [r[1] for r in d['recon_err_log']], label=k)
ax[0].set(xlabel='step', ylabel='train loss', title='Loss trajectory'); ax[0].legend()
ax[1].set(xlabel='step', ylabel='relative recon error', title='Activation reconstruction error')
ax[1].legend(); plt.tight_layout(); plt.savefig('results/curves.png', dpi=120); plt.show()
"""),
    md("Pick the winner on **loss trajectory first, reconstruction error second, speed third** — "
       "a fast variant with a diverging inverse is training on wrong gradients."),
])

N3 = nb([
    md("""
# Session 13 — 3/3: push the batch, scan depth, build the report

Run 3 takes the winning reversible variant and pushes the batch size to the
edge of the GPU. Then a depth scan isolates the actual claim of reversibility:
activation memory that does not grow with the number of layers.
"""),
    code(SETUP),
    md("## Max batch for each mode"),
    code("!python src/train.py --mode midpoint --find_max_batch --seq_len 512\n"
         "!python src/train.py --mode euler    --find_max_batch --seq_len 512"),
    md("## Run 3 — midpoint @ max batch"),
    code("import json\n"
         "BATCH_MAX = json.load(open('results/maxbatch_midpoint.json'))['max_batch']\n"
         "BATCH_FIX = json.load(open('results/maxbatch_baseline.json'))['max_batch']\n"
         "print(f'baseline max {BATCH_FIX} -> midpoint max {BATCH_MAX} "
         "({BATCH_MAX/BATCH_FIX:.2f}x)')\n"
         "!python src/train.py --mode midpoint --batch_size {BATCH_MAX} --run_name midpoint_max \\\n"
         "    --h 0.5 --check_recon --seq_len 512 --total_tokens 50000000"),
    md("## Memory vs depth\n\nSame batch and sequence length, layers swept. Baseline should grow "
       "roughly linearly; midpoint should stay flat."),
    code("!python scripts/depth_scan.py --layers 4 8 16 32 64 --batch_size 8 --seq_len 512"),
    md("## Cost comparison\n\nWhat the memory saving is worth in rupees/dollars: slower per token, "
       "but a smaller GPU tier or fewer nodes."),
    code("""
import json
b = json.load(open('results/baseline_fixed.json'))
m = json.load(open('results/midpoint_fixed.json'))
PRICE_PER_HR = 0.35   # <-- set to the actual price of the GPU tier you used
for name, d in [('baseline', b), ('midpoint', m)]:
    hrs = d['tokens_trained'] / d['tok_per_s'] / 3600
    print(f"{name:9s} {hrs:.2f} GPU-hours  ${hrs*PRICE_PER_HR:.2f} for "
          f"{d['tokens_trained']:,} tokens  (peak {d['peak_mem_gb']:.2f} GB)")
print('\\nreversibility costs %.1f%% more compute-time for %.2fx the peak memory'
      % (100*(b['tok_per_s']/m['tok_per_s']-1), m['peak_mem_gb']/b['peak_mem_gb']))
"""),
    md("## Generate the README and verify it"),
    code("!python scripts/make_readme.py\n"
         "!pytest tests/ -q"),
    code("!git add -A && git -c user.email=rajivs.iitkgp@gmail.com -c user.name=rjvim \\\n"
         "    commit -q -m 'session 13: results + generated README' && git push"),
])

os.makedirs(NB, exist_ok=True)
for name, n in [("01_setup_and_baseline", N1), ("02_reversible_variants", N2),
                ("03_max_batch_and_report", N3)]:
    with open(os.path.join(NB, name + ".ipynb"), "w") as f:
        json.dump(n, f, indent=1)
    print("wrote", name)
