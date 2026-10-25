"""Termination terms for DexReMoE.

  object_dropped : z drops more than 2 * size below reset z
  object_far     : distance from reset pos exceeds 3 * size
  max_success    : consecutive_success >= cap (max_consecutive_success)

`time_out` is the standard term from `isaaclab.envs.mdp`.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .commands import DexReMoECommand


_DROP_FACTOR = 2.0
_FAR_FACTOR = 3.0


def object_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj = env.active_root_state_w()
    return obj["pos"][:, 2] < (env.object_pos_init[:, 2] - _DROP_FACTOR * env.object_size_m)


def object_far(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj = env.active_root_state_w()
    dist = torch.norm(obj["pos"] - env.object_pos_init, dim=-1)
    return dist > _FAR_FACTOR * env.object_size_m


def max_consecutive_success_reached(env: ManagerBasedRLEnv, cap: float = 25.0) -> torch.Tensor:
    cmd: DexReMoECommand = env.command_manager.get_term("object_pose")
    return cmd.get_max_consecutive_success() >= cap
