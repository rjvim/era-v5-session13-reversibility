"""Activation-free execution of a stack of residual functions.

Two integrators, both from the session-13 material:

  euler     s_{l+1} = s_l     + h  * f_l(s_l)
            inverse: s_l  <- s_{l+1} - h * f_l(s_l)   (fixed-point iteration;
            a contraction iff h * Lip(f_l) < 1, so it is EXACT ONLY IN THE
            LIMIT and can silently diverge)

  midpoint  s_{l+1} = s_{l-1} + 2h * f_l(s_l)         (leapfrog)
            inverse: s_{l-1} = s_{l+1} - 2h * f_l(s_l)  -- CLOSED FORM, exact
            up to floating point, no iteration, no assumption on Lip(f).

Forward runs under no_grad and keeps only the boundary states, so activation
memory is O(1) in depth instead of O(n_layer).  Backward walks down the stack,
rebuilding each s_l and doing a local forward+vjp for that layer only.
"""
from __future__ import annotations

import torch


def _autocast_dtype(dev):
    """torch >= 2.4 has get_autocast_dtype; older builds have per-device getters."""
    if hasattr(torch, "get_autocast_dtype"):
        return torch.get_autocast_dtype(dev)
    return (torch.get_autocast_gpu_dtype() if dev == "cuda"
            else torch.get_autocast_cpu_dtype())


def _rel_err(rebuilt, truth):
    """Relative reconstruction error: max|rebuilt - truth| / max|truth|."""
    denom = truth.abs().max().item()
    return (rebuilt - truth).abs().max().item() / (denom + 1e-12)


def _params_of(layers):
    ps, slices = [], []
    i = 0
    for layer in layers:
        p = [q for q in layer.parameters() if q.requires_grad]
        ps.extend(p)
        slices.append((i, i + len(p)))
        i += len(p)
    return ps, slices


class _RevStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layers, mode, h, euler_iters, check_recon, n_params, *params):
        # The backward pass RE-RUNS layers. It must re-run them under the same
        # autocast state as the forward, or the rebuilt activations are a
        # different dtype from the ones they are replacing -- which silently
        # skews both the gradients and the measured throughput.
        ctx.amp = {
            "device_type": "cuda" if x.is_cuda else "cpu",
            "enabled": torch.is_autocast_enabled("cuda" if x.is_cuda else "cpu"),
            "dtype": _autocast_dtype("cuda" if x.is_cuda else "cpu"),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        ctx.layers = layers
        ctx.mode = mode
        ctx.h = h
        ctx.euler_iters = euler_iters
        ctx.check_recon = check_recon
        L = len(layers)

        with torch.no_grad():
            if mode == "euler":
                s = x
                for l in range(L):
                    s = s + h * layers[l](s)
                out = s
                ctx.save_for_backward(x, out.detach())
                ctx.s_prev = None
            elif mode == "midpoint":
                s_prev = x                                  # s_0
                s_curr = x + h * layers[0](x)               # s_1 (Euler bootstrap)
                for l in range(1, L):
                    s_next = s_prev + 2.0 * h * layers[l](s_curr)
                    s_prev, s_curr = s_curr, s_next
                out = s_curr                                # s_L
                ctx.save_for_backward(x, out.detach(), s_prev.detach())
            else:
                raise ValueError(mode)
        return out.detach()

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _local_vjp(layer, s, cot):
        """One layer forward with grad + vjp. Returns (f(s), grad_s, param_grads)."""
        with torch.enable_grad():
            si = s.detach().requires_grad_(True)
            fo = layer(si)
        ps = [q for q in layer.parameters() if q.requires_grad]
        grads = torch.autograd.grad(fo, [si] + ps, grad_outputs=cot, allow_unused=True)
        return fo.detach(), grads[0], list(grads[1:])

    @staticmethod
    def backward(ctx, grad_out):
        with torch.autocast(**ctx.amp):
            return _RevStack._backward_impl(ctx, grad_out)

    @staticmethod
    def _backward_impl(ctx, grad_out):
        layers, mode, h = ctx.layers, ctx.mode, ctx.h
        L = len(layers)
        all_params, slices = _params_of(layers)
        pgrads = [None] * len(all_params)

        def stash(l, gs):
            a, b = slices[l]
            for k, g in enumerate(gs):
                if g is None:
                    continue
                pgrads[a + k] = g if pgrads[a + k] is None else pgrads[a + k] + g

        recon_err = float("nan")

        if mode == "euler":
            x0, s_hi = ctx.saved_tensors
            g = grad_out
            for l in range(L - 1, -1, -1):
                # rebuild s_l from s_{l+1} by fixed-point iteration
                with torch.no_grad():
                    z = s_hi
                    for _ in range(ctx.euler_iters):
                        z = s_hi - h * layers[l](z)
                s_lo = z
                _, gs, gp = _RevStack._local_vjp(layers[l], s_lo, h * g)
                stash(l, gp)
                g = g + gs
                s_hi = s_lo
            if ctx.check_recon:
                recon_err = _rel_err(s_hi, x0)
            grad_in = g
        else:  # midpoint
            x0, s_L, s_Lm1 = ctx.saved_tensors
            s_hi, s_lo = s_L, s_Lm1          # (s_{l+1}, s_l) with l = L-1
            gp_, gc_ = grad_out, torch.zeros_like(grad_out)   # g_{l+1}, g_l
            for l in range(L - 1, 0, -1):
                fo, gs, gpar = _RevStack._local_vjp(layers[l], s_lo, 2.0 * h * gp_)
                stash(l, gpar)
                gc_ = gc_ + gs                      # g_l complete
                g_lm1 = gp_                         # identity path s_{l+1} = s_{l-1} + ...
                with torch.no_grad():
                    s_new = s_hi - 2.0 * h * fo     # s_{l-1}
                s_hi, s_lo = s_lo, s_new
                gp_, gc_ = gc_, g_lm1
            # bootstrap layer 0: s_1 = s_0 + h f_0(s_0)
            _, gs0, gpar0 = _RevStack._local_vjp(layers[0], s_lo, h * gp_)
            stash(0, gpar0)
            grad_in = gc_ + gp_ + gs0
            if ctx.check_recon:
                recon_err = _rel_err(s_lo, x0)

        ctx.recon_err = recon_err
        _LAST["recon_err"] = recon_err
        return (grad_in, None, None, None, None, None, None, *pgrads)


_LAST = {"recon_err": float("nan")}


def last_recon_err():
    """Reconstruction error of the most recent backward pass (max abs, input state)."""
    return _LAST["recon_err"]


def reversible_stack(x, layers, mode="midpoint", h=0.5, euler_iters=8, check_recon=False):
    params, _ = _params_of(layers)
    out = _RevStack.apply(x, layers, mode, h, euler_iters, check_recon, len(params), *params)
    return out, _LAST["recon_err"]


# ---------------------------------------------------------------------------
# Reference (memory-hungry) implementations, used only by the tests to verify
# that the custom backward reproduces ordinary autograd exactly.
# ---------------------------------------------------------------------------
def reference_stack(x, layers, mode="midpoint", h=0.5):
    L = len(layers)
    if mode == "euler":
        s = x
        for l in range(L):
            s = s + h * layers[l](s)
        return s
    if mode == "midpoint":
        s_prev = x
        s_curr = x + h * layers[0](x)
        for l in range(1, L):
            s_prev, s_curr = s_curr, s_prev + 2.0 * h * layers[l](s_curr)
        return s_curr
    raise ValueError(mode)
