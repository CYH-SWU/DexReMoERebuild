"""mu_e: e_t -> z_t.

Input : (N, 55) = [privileged_norm(23), f_t(32)]
Output: (N, 66) — LayerNorm-ed privileged embedding `z_t`.

The mapping is an expansion (55 -> 66), not a compression. The value of
mu_e is fusing `e_phys` + size one-hot + shape feature into a form the
policy/critic can consume directly, matching the paper's convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MuE(nn.Module):
    """Single-hidden-layer MLP followed by LayerNorm."""

    def __init__(self, in_dim: int = 55, hidden: int = 128, out_dim: int = 66):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        return self.net(e_t)
