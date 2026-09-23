"""GPT backbone shared by all three arms.

The *architecture* (block internals, parameter count) is identical across
arms. Only the inter-layer update rule (the "integrator") changes:

  baseline : x_{l+1} = x_l + f_l(x_l)                 (stored activations)
  euler    : x_{l+1} = x_l + h * f_l(x_l)             (invert by fixed point)
  midpoint : x_{l+1} = x_{l-1} + 2h * f_l(x_l)        (invert exactly)

This keeps the comparison honest: same weights, same FLOPs per block, only
memory strategy and integrator differ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from reversible import last_recon_err, reversible_stack


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 256
    mode: str = "baseline"  # baseline | euler | midpoint
    h: float = 0.5  # integrator step size (unused by baseline)
    euler_iters: int = 8  # fixed-point iterations for euler inversion
    dropout: float = 0.0  # MUST stay 0 for reversible arms (see README)

    def dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class ResidualFn(nn.Module):
    """f_l(x): the pure update function used by every integrator.

    f(x) = [a + mlp(ln2(a))] - x    where a = x + attn(ln1(x))

    baseline then computes x + f(x), which is exactly the usual pre-LN block.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        a = self.attn(self.ln_1(x))
        b = self.mlp(self.ln_2(x + a))
        return a + b


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.mode in ("baseline", "euler", "midpoint")
        if cfg.mode != "baseline":
            assert cfg.dropout == 0.0, "reversible arms forbid dropout"
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.layers = nn.ModuleList([ResidualFn(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @property
    def last_recon_err(self):
        """Valid after .backward(); nan for the baseline arm."""
        return last_recon_err()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, check_recon=False):
        B, T = idx.shape
        cfg = self.cfg
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]

        if cfg.mode == "baseline":
            for blk in self.layers:
                x = x + blk(x)
        else:
            x, _ = reversible_stack(
                x, self.layers, mode=cfg.mode, h=cfg.h,
                euler_iters=cfg.euler_iters, check_recon=check_recon,
            )

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
