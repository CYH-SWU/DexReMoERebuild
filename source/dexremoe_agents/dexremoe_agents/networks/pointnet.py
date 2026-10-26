"""PointNet encoder for unit-scale object point clouds.

Input : (N, 512, 3) — per-point xyz in the object's local frame, already
        normalized to the unit cube/sphere/cylinder.
Output: (N, 32)     — LayerNorm-ed shape feature `f_t`.

We use vanilla PointNet (not PointNet++) because the objects are three
convex primitives (cuboid / cylinder / sphere); hierarchical local feature
learning would add compute without signal. The output interface matches
the paper's `f_t` (32-D), so it can be swapped for PointNet++ later without
touching downstream code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNet(nn.Module):
    """Per-point MLP + global max pool + head MLP + LayerNorm."""

    def __init__(
        self,
        in_dim: int = 3,
        point_hidden: tuple[int, ...] = (64, 128, 512),
        head_hidden: tuple[int, ...] = (512, 256),
        out_dim: int = 32,
    ):
        super().__init__()
        # --- per-point MLP (Conv1d over the point dimension) ---
        layers: list[nn.Module] = []
        c_in = in_dim
        for c_out in point_hidden:
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=1))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ELU(inplace=True))
            c_in = c_out
        self.point_mlp = nn.Sequential(*layers)

        # --- head MLP on the pooled global feature ---
        head: list[nn.Module] = []
        c_in = point_hidden[-1]
        for c_out in head_hidden:
            head.append(nn.Linear(c_in, c_out))
            head.append(nn.ELU(inplace=True))
            c_in = c_out
        head.append(nn.Linear(c_in, out_dim))
        head.append(nn.LayerNorm(out_dim))
        self.head = nn.Sequential(*head)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (N, P, 3) -> (N, 3, P) for Conv1d
        x = points.transpose(1, 2)
        x = self.point_mlp(x)                     # (N, 1024, P)
        x = x.max(dim=2).values                   # (N, 1024)
        return self.head(x)                       # (N, 32), LayerNorm-ed
