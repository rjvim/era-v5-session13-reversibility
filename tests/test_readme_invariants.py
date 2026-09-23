"""Every number the README states must match the JSON that produced it.

This is the self-checking pattern: the README is generated from results/, and
this test re-parses the rendered README and asserts it against results/ again.
A stale or hand-edited README fails CI instead of quietly misreporting.
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")

ARMS = ["baseline_fixed", "euler_fixed", "midpoint_fixed", "midpoint_max"]


def _runs():
    out = {}
    for a in ARMS:
        p = os.path.join(RESULTS, f"{a}.json")
        if os.path.exists(p):
            out[a] = json.load(open(p))
    return out


def _need_runs():
    if not os.path.exists(README) or not _runs():
        pytest.skip("no rendered README / no training results yet")


def _rows():
    """Parse the main results table out of the README."""
    text = open(README).read()
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|") or "`" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 9:
            continue
        rows[cells[0]] = cells
    return rows


LABEL = {
    "baseline_fixed": "1. Baseline (no reversibility)",
    "euler_fixed": "2a. Reversible - euler",
    "midpoint_fixed": "2b. Reversible - midpoint",
    "midpoint_max": "3. Reversible - midpoint @ max batch",
}


@pytest.mark.parametrize("arm", ARMS)
def test_table_matches_json(arm):
    _need_runs()
    runs = _runs()
    if arm not in runs:
        pytest.skip(f"{arm} not run yet")
    d = runs[arm]
    row = _rows().get(LABEL[arm])
    assert row is not None, f"no README row for {arm}"
    assert row[1].strip("`") == d["mode"]
    assert int(row[2]) == d["batch_size"]
    assert float(row[4]) == pytest.approx(d["final_train_loss"], abs=5e-5)
    assert float(row[5]) == pytest.approx(d["final_val_loss"], abs=5e-5)
    assert float(row[6].replace(",", "")) == pytest.approx(d["tok_per_s"], rel=1e-3)


def test_every_run_used_the_full_token_budget():
    _need_runs()
    for name, d in _runs().items():
        assert d["tokens_trained"] >= 0.98 * d["total_steps"] * d["tokens_per_step"], name
        assert d["tokens_trained"] >= 49_000_000 or d["config"]["vocab_size"] < 1000, name


def test_arms_share_architecture_data_and_optimiser():
    _need_runs()
    runs = _runs()
    if len(runs) < 2:
        pytest.skip("need >= 2 runs")
    keys = ("n_layer", "n_head", "n_embd", "vocab_size", "block_size")
    ref = runs[ARMS[0]] if ARMS[0] in runs else list(runs.values())[0]
    for name, d in runs.items():
        for k in keys:
            assert d["config"][k] == ref["config"][k], f"{name}.{k}"
        assert d["seed"] == ref["seed"], name
        assert d["config"]["dropout"] == 0.0, name
        assert d["params_total"] == ref["params_total"], name


def test_reversible_runs_recorded_reconstruction_error():
    _need_runs()
    for name, d in _runs().items():
        if d["mode"] == "baseline":
            continue
        assert d["max_recon_err"] is not None, f"{name}: run with --check_recon"


def test_headline_claims_are_consistent_with_results():
    _need_runs()
    runs = _runs()
    if not {"baseline_fixed", "midpoint_fixed"} <= runs.keys():
        pytest.skip("need baseline + midpoint")
    text = open(README).read()
    b, m = runs["baseline_fixed"], runs["midpoint_fixed"]
    claimed = re.search(r"costs \*\*([\d.]+)x\*\* of baseline throughput", text)
    assert claimed, "headline speed claim missing"
    assert float(claimed.group(1)) == pytest.approx(m["tok_per_s"] / b["tok_per_s"], abs=5e-3)


def test_precision_probe_supports_the_variant_claim():
    """The README's 'midpoint inverse is exact, euler's is not' claim must hold
    in the artifact it cites."""
    p = os.path.join(RESULTS, "precision_probe.json")
    if not os.path.exists(p):
        pytest.skip("probe not run")
    d = json.load(open(p))
    rows = {r["scale"]: r for r in d["fp32_vs_weight_scale"]}
    assert rows[1]["midpoint"] < 1e-5 and rows[1]["euler"] < 1e-5   # both fine when tame
    for s in (5, 20, 40):
        assert rows[s]["euler"] > 100 * rows[s]["midpoint"], s      # euler diverges first
    it = d["fp32_euler_vs_iterations"]
    assert it[-1]["euler"] > 1.0, "more iterations should not rescue a non-contraction"


def test_activation_memory_is_flat_in_depth():
    """The central claim: reversible activation memory does not grow with depth."""
    p = os.path.join(RESULTS, "saved_bytes_scan.json")
    if not os.path.exists(p):
        pytest.skip("scan not run")
    rows = json.load(open(p))["rows"]
    lo, hi = rows[0], rows[-1]
    assert hi["baseline_bytes"] > 10 * lo["baseline_bytes"]      # baseline grows
    assert hi["midpoint_bytes"] == pytest.approx(lo["midpoint_bytes"], rel=0.15)
    assert hi["baseline_bytes"] / hi["midpoint_bytes"] > 20


def test_cpu_pilot_report_matches_its_results():
    """CPU_PILOT.md must agree with results_cpu/, same rule as the main README."""
    rc = os.path.join(ROOT, "results_cpu")
    md = os.path.join(ROOT, "CPU_PILOT.md")
    if not (os.path.isdir(rc) and os.path.exists(md)):
        pytest.skip("no CPU pilot")
    text = open(md).read()
    for arm in ("baseline_fixed", "euler_fixed", "midpoint_fixed", "midpoint_max"):
        p = os.path.join(rc, f"{arm}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        assert f"{d['final_val_loss']:.4f}" in text, arm
        assert f"{d['tok_per_s']:,.0f}" in text, arm


def test_cpu_pilot_euler_reconstruction_actually_broke():
    """The headline pilot finding: euler's inverse fails in real training while
    its loss curve looks normal. If this ever stops holding, the claim goes."""
    rc = os.path.join(ROOT, "results_cpu")
    if not os.path.isdir(rc):
        pytest.skip("no CPU pilot")
    e = os.path.join(rc, "euler_fixed.json")
    m = os.path.join(rc, "midpoint_fixed.json")
    if not (os.path.exists(e) and os.path.exists(m)):
        pytest.skip("pilot incomplete")
    de, dm = json.load(open(e)), json.load(open(m))
    assert de["max_recon_err"] > 1e3 * dm["max_recon_err"]
    assert abs(de["final_val_loss"] - dm["final_val_loss"]) < 0.2   # loss hides it
