"""Reward terms for DexReMoE.

r1 = +800 on just_succeeded (one-shot)
r2 = -10 * ||pos_diff|| + 1/(||dtheta|| + 0.1)
r3 = -0.0002 * sum_i max(|w_i| - 6, 0)
     - 0.0001 * ||a_t||^2
     - 0.01   * ||a_t - a_{t-1}||^2
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .commands import DexReMoECommand


# --- 1. success bonus ---

def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-shot +1 when the command term reports a fresh success (scaled by cfg weight)."""
    cmd: DexReMoECommand = env.command_manager.get_term("object_pose")
    return cmd.just_succeeded.float()


# --- 2. pose track ---

def orientation_inv_track(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 / (dtheta + eps) - encourages approaching the goal."""
    obj = env.active_root_state_w()
    cmd: DexReMoECommand = env.command_manager.get_term("object_pose")
    dtheta = math_utils.quat_error_magnitude(obj["quat"], cmd.quat_command_w)
    return 1.0 / (dtheta + 0.1)


def position_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Negative L2 distance between object pos and the pinned position goal."""
    obj = env.active_root_state_w()
    cmd: DexReMoECommand = env.command_manager.get_term("object_pose")
    return torch.norm(obj["pos"] - cmd.pos_command_w, dim=-1)


# --- 3. control cost ---

def joint_vel_clipped_penalty(env: ManagerBasedRLEnv, clip: float = 6.0,
                              robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum over joints of max(|w_i| - clip, 0)."""
    robot = env.scene[robot_cfg.name]
    return torch.clamp(robot.data.joint_vel.abs() - clip, min=0.0).sum(dim=-1)
