"""Running mean/variance normalizer for input-side quantities.

Use this only for signals whose distribution is fixed by the env (e.g.
policy_obs, continuous privileged features). Do NOT apply it to network
intermediate outputs (`f_t`, `z_t`) — their distributions shift as training
progresses, which would cause the stats to lag and couple gradients.

Stats are registered as buffers, so `state_dict()` / `load_state_dict()`
transport them together with the network weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Tracks running mean/std of a fixed-dim tensor and normalizes it.

    Args:
        dim: feature dimension of the input (last dim of the tensor).
        eps: minimum variance floor for the denominator.
        update: if True (default), running stats are updated on forward when
            the module is in training mode. If False, stats stay fixed.
    """

    def __init__(self, dim: int, eps: float = 1e-5, update: bool = True):
        super().__init__()
        self.eps = eps
        self.update = update
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        # x: (N, dim)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / new_count
        new_var = (
            self.var * self.count
            + batch_var * batch_count
            + delta.pow(2) * self.count * batch_count / new_count
        ) / new_count
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RSL-RL wraps the whole rollout in torch.inference_mode(). Stats must
        # not be updated there: the resulting inference tensors would leak
        # into the running buffers and break on the first autograd pass.
        if self.training and self.update and not torch.is_inference_mode_enabled():
            self._update_stats(x.detach())
        return (x - self.mean) / torch.sqrt(self.var + self.eps)
