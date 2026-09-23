"""Report for the CPU pilot in results_cpu/.

This is NOT the assignment's 50M-token GPU study. It is a smaller, fully
measured run used to (a) prove the pipeline end to end and (b) establish the
loss / speed / reconstruction relationships before spending GPU time.
Every number here was produced on 1 CPU core and is labelled as such.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results_cpu")
ARMS = [("baseline_fixed", "Baseline"), ("euler_fixed", "Reversible - euler"),
        ("midpoint_fixed", "Reversible - midpoint"),
        ("midpoint_max", "Reversible - midpoint @ 2x batch")]


def load(n):
    p = os.path.join(R, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    runs = {k: load(k) for k, _ in ARMS}
    have = {k: v for k, v in runs.items() if v}
    if not have:
        print("no results_cpu yet"); return
    ref = list(have.values())[0]
    rows = ["| Arm | Batch | Final train loss | Val loss | tok/s | Max recon err |",
            "|---|---|---|---|---|---|"]
    for k, label in ARMS:
        d = runs.get(k)
        if not d:
            rows.append(f"| {label} | -- | -- | -- | -- | -- |"); continue
        re_ = d["max_recon_err"]
        rows.append(f"| {label} | {d['batch_size']} | {d['final_train_loss']:.4f} | "
                    f"{d['final_val_loss']:.4f} | {d['tok_per_s']:,.0f} | "
                    f"{re_:.2e} |" if re_ else
                    f"| {label} | {d['batch_size']} | {d['final_train_loss']:.4f} | "
                    f"{d['final_val_loss']:.4f} | {d['tok_per_s']:,.0f} | n/a |")
    b, m, e = runs.get("baseline_fixed"), runs.get("midpoint_fixed"), runs.get("euler_fixed")
    lines = [
        "# CPU pilot — reversibility, measured end to end",
        "",
        "> **Scope.** 1 CPU core, fp32, tiny-shakespeare (GPT-2 BPE remapped to the "
        f"{ref['config']['vocab_size']:,} ids that actually occur), "
        f"{ref['config']['n_layer']} layers / {ref['config']['n_embd']} d_model / "
        f"seq {ref['config']['block_size']}, {ref['tokens_trained']:,} tokens per arm. "
        "This is a pilot, not the assignment's 20M-param / 50M-token GPU study — it exists "
        "to verify the pipeline and establish the relationships before spending GPU hours. "
        "Peak-memory columns are GPU-only and are therefore absent here; activation memory "
        "is measured device-independently in `results/saved_bytes_scan.json` instead.",
        "", "\n".join(rows), "",
        "## What this pilot already establishes", "",
    ]
    if b and m:
        lines.append(f"1. **Midpoint costs {100*(b['tok_per_s']/m['tok_per_s']-1):.0f}% throughput** "
                     f"({b['tok_per_s']:,.0f} -> {m['tok_per_s']:,.0f} tok/s) for a val-loss "
                     f"difference of {m['final_val_loss']-b['final_val_loss']:+.4f}.")
    if b and e:
        lines.append(f"2. **Euler costs {100*(b['tok_per_s']/e['tok_per_s']-1):.0f}% throughput** "
                     f"({e['tok_per_s']:,.0f} tok/s) for val loss "
                     f"{e['final_val_loss']-b['final_val_loss']:+.4f} vs baseline — the extra "
                     f"cost is its fixed-point inversion, not extra modelling capacity.")
    if m and e and m["max_recon_err"] and e["max_recon_err"]:
        lines.append(f"3. **Reconstruction error during real training:** midpoint "
                     f"{m['max_recon_err']:.2e} vs euler {e['max_recon_err']:.2e} "
                     f"({e['max_recon_err']/m['max_recon_err']:.0f}x worse).")
    mm = runs.get("midpoint_max")
    if mm and m:
        lines.append(f"4. **Doubling the batch** (the memory headroom reversibility buys) moved "
                     f"val loss from {m['final_val_loss']:.4f} to {mm['final_val_loss']:.4f} at "
                     f"the same token budget — half as many optimiser steps.")
    with open(os.path.join(ROOT, "CPU_PILOT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote CPU_PILOT.md")


if __name__ == "__main__":
    main()
