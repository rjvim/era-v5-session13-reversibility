"""Correctness invariants for the reversible stack.

These run on CPU in seconds and are what makes the memory claims trustworthy:
if the custom backward did not reproduce ordinary autograd, the memory saving
would be meaningless.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import GPT, GPTConfig, ResidualFn  # noqa: E402
from reversible import reference_stack, reversible_stack  # noqa: E402

torch.manual_seed(0)
CFG = GPTConfig(vocab_size=128, block_size=16, n_layer=6, n_head=2, n_embd=32)


def _layers(cfg=CFG):
    torch.manual_seed(1)
    return torch.nn.ModuleList([ResidualFn(cfg) for _ in range(cfg.n_layer)]).double()


def _grads(layers, x, fn):
    for p in layers.parameters():
        p.grad = None
    xi = x.clone().requires_grad_(True)
    out = fn(xi, layers)
    loss = (out ** 2).mean()
    loss.backward()
    return loss.item(), xi.grad.clone(), [p.grad.clone() for p in layers.parameters()]


@pytest.mark.parametrize("mode", ["midpoint", "euler"])
def test_custom_backward_matches_autograd(mode):
    h = 0.5 if mode == "midpoint" else 0.1
    layers = _layers()
    x = torch.randn(2, CFG.block_size, CFG.n_embd, dtype=torch.float64)

    l_ref, gx_ref, gp_ref = _grads(layers, x, lambda a, L: reference_stack(a, L, mode, h))
    l_rev, gx_rev, gp_rev = _grads(
        layers, x, lambda a, L: reversible_stack(a, L, mode, h, euler_iters=40)[0]
    )

    assert abs(l_ref - l_rev) < 1e-10
    assert torch.allclose(gx_ref, gx_rev, atol=1e-8, rtol=1e-6)
    for a, b in zip(gp_ref, gp_rev):
        assert torch.allclose(a, b, atol=1e-8, rtol=1e-6)


def test_midpoint_reconstruction_is_exact_fp32():
    cfg = GPTConfig(**{**CFG.dict(), "mode": "midpoint", "h": 0.5})
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = m(idx, idx, check_recon=True)
    loss.backward()
    assert m.last_recon_err < 1e-4, m.last_recon_err


def _recon_err(mode, h, scale, iters=8, seed=3):
    cfg = GPTConfig(**{**CFG.dict(), "mode": mode, "h": h, "euler_iters": iters})
    torch.manual_seed(seed)
    m = GPT(cfg)
    with torch.no_grad():                      # amplify f so that h*Lip(f) > 1
        for p in m.layers.parameters():
            p.mul_(scale)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = m(idx, idx, check_recon=True)
    loss.backward()
    return m.last_recon_err


def test_euler_inversion_breaks_when_not_a_contraction():
    """Honest finding: euler inversion is a fixed point, not a closed form.
    Once h * Lip(f) exceeds 1 the iteration stops converging and the
    reconstructed activations -- hence the gradients -- are wrong."""
    tame = _recon_err("euler", 0.5, scale=1.0)
    wild = _recon_err("euler", 0.5, scale=40.0)
    assert tame < 1e-5
    assert wild > 1e3


def test_midpoint_inversion_survives_the_same_stress():
    """Same amplified weights: the closed-form inverse stays usable where the
    euler fixed point has already diverged by ~6 orders of magnitude."""
    mid = _recon_err("midpoint", 0.5, scale=40.0)
    eul = _recon_err("euler", 0.5, scale=40.0)
    assert mid < 1e-1
    assert eul > 1e3 * mid


def test_all_arms_have_identical_parameter_counts():
    counts = set()
    for mode in ("baseline", "euler", "midpoint"):
        cfg = GPTConfig(**{**CFG.dict(), "mode": mode})
        counts.add(GPT(cfg).num_params())
    assert len(counts) == 1


def test_reversible_arms_reject_dropout():
    with pytest.raises(AssertionError):
        GPT(GPTConfig(**{**CFG.dict(), "mode": "midpoint", "dropout": 0.1}))


def test_forward_does_not_retain_intermediate_activations():
    """The saved-tensor count must not grow with depth for reversible arms."""
    def saved_count(mode, n_layer):
        cfg = GPTConfig(**{**CFG.dict(), "mode": mode, "n_layer": n_layer})
        m = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        seen = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (seen.append(t.numel()), t)[1], lambda t: t
        ):
            _, loss = m(idx, idx)
        return sum(seen)

    rev_small, rev_big = saved_count("midpoint", 2), saved_count("midpoint", 16)
    base_small, base_big = saved_count("baseline", 2), saved_count("baseline", 16)
    assert base_big > 3 * base_small          # baseline grows with depth
    assert rev_big < 1.15 * rev_small         # reversible is flat in depth


def test_backward_recompute_respects_autocast():
    """Regression: the backward pass re-runs layers, and it must do so under
    the SAME autocast state as the forward. Running the recompute in fp32 while
    the forward ran in bf16 produces gradients that are wrong by ~1e-2 relative
    -- small enough to be mistaken for bf16 noise."""
    layers = _layers().float()
    x = torch.randn(2, CFG.block_size, CFG.n_embd)

    def run(fn):
        for p in layers.parameters():
            p.grad = None
        xi = x.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = fn(xi, layers)
            loss = (out.float() ** 2).mean()
        loss.backward()
        return [p.grad.clone() for p in layers.parameters()]

    ref = run(lambda a, L: reference_stack(a, L, "midpoint", 0.5))
    rev = run(lambda a, L: reversible_stack(a, L, "midpoint", 0.5)[0])
    worst = max(((a - b).abs().max() / (a.abs().max() + 1e-9)).item()
                for a, b in zip(ref, rev))
    assert worst < 1e-5, f"autocast state not propagated into backward: {worst:.2e}"
