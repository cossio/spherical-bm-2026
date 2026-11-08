# /// script
# requires-python = ">=3.14"
# dependencies = ["torch"]
# ///
"""FVSBN student for +/-1 spins.

Verbatim copy of ``BinaryFVSBN`` from
``double_descent/ising2d_van/run.py``; kept local so this experiment is
self-contained inside ``05_fvsbn_cw/``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFVSBN(nn.Module):
    """Fully-visible sigmoid belief net: q(s_i=+1|s_{<i}) = sigmoid(b_i + W_i . s_{<i})."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.W = nn.Parameter(torch.zeros(N, N))
        self.b = nn.Parameter(torch.zeros(N))
        self.register_buffer("mask", torch.tril(torch.ones(N, N), diagonal=-1))

    def _logits(self, s: torch.Tensor) -> torch.Tensor:
        return s @ (self.W * self.mask).T + self.b

    def log_prob(self, s: torch.Tensor) -> torch.Tensor:
        logits = self._logits(s)
        target = (s + 1.0) * 0.5
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)

    @torch.no_grad()
    def sample_with_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        W_eff = (self.W * self.mask).detach()
        b = self.b.detach()
        s = torch.zeros(n, self.N)
        log_q = torch.zeros(n)
        for i in range(self.N):
            logit_i = s @ W_eff[i] + b[i]
            p_up = torch.sigmoid(logit_i)
            s_i = torch.where(torch.rand(n) < p_up, 1.0, -1.0)
            s[:, i] = s_i
            log_q += torch.where(s_i > 0, F.logsigmoid(logit_i), F.logsigmoid(-logit_i))
        return s, log_q
