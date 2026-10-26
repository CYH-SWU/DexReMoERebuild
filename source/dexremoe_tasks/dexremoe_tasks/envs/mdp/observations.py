"""Observation terms for DexReMoE.

All position-like quantities are differences of world positions (origin
cancels). Rotations are quaternion relative to goal. Velocities are world
frame. No absolute world position is exposed anywhere.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .commands import DexReMoECommand


# --------------------------------------------------------------- policy (96)


def policy_obs(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """3-step joint position history + 3-step last action history.

    Note: history is handled by the observation manager (`history_length=3`
    per term). This function returns the *current* 32-dim frame; the manager
    concatenates 3 frames to make 96.
    """
    robot = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos  # (N, 16)
    last_action = env.action_manager.action  # (N, 16)
    return torch.cat([joint_pos, last_action], dim=-1)  # (N, 32)


# --------------------------------------------------------------- privileged (22)


def _size_bucket_one_hot(scale: torch.Tensor) -> torch.Tensor:
    """4-dim bucket one-hot. Boundaries 0.74/0.78/0.82 with float32 eps."""
    eps = 1e-4
    b0 = (scale <= 0.74 + eps).float()
    b1 = ((scale > 0.74 + eps) & (scale <= 0.78 + eps)).float()
    b2 = ((scale > 0.78 + eps) & (scale <= 0.82 + eps)).float()
    b3 = (scale > 0.82 + eps).float()
    return torch.stack([b0, b1, b2, b3], dim=-1)


def privileged_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """e_phys (19) + size_bucket_one_hot (4) = 23.

    Shape is NOT encoded here: the PointNet is the sole source of shape
    information. Size appears twice on purpose: `scale` is the continuous
    value inside e_phys, and the 4-dim bucket one-hot gives a discrete cue.
    """
    obj = env.active_root_state_w()
    cmd: DexReMoECommand = env.command_manager.get_term("object_pose")

    mass = env.active_mass().unsqueeze(-1)                                     # 1
    friction = torch.ones(env.num_envs, 1, device=env.device)                  # 1 (constant)
    scale = env.active_scale().unsqueeze(-1)                                   # 1
    com = torch.zeros(env.num_envs, 3, device=env.device)                      # 3 (constant)
    pos_diff = obj["pos"] - cmd.pos_command_w                                  # 3
    quat_diff = math_utils.quat_mul(obj["quat"], math_utils.quat_conjugate(cmd.quat_command_w))
    quat_diff = math_utils.quat_unique(quat_diff)                              # 4
    lin_vel = obj["lin_vel"]                                                   # 3
    ang_vel = obj["ang_vel"]                                                   # 3
    e_phys = torch.cat([mass, friction, scale, com, pos_diff, quat_diff, lin_vel, ang_vel], dim=-1)

    size_one_hot = _size_bucket_one_hot(env.active_scale())                    # 4

    return torch.cat([e_phys, size_one_hot], dim=-1)  # (N, 23)


# --------------------------------------------------------------- size_encoding (5)


def size_encoding_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[scale, b0?, b1?, b2?, b3?] with b0 lower-open, b3 upper-open. Gate only."""
    scale = env.active_scale()
    one_hot = _size_bucket_one_hot(scale)
    return torch.cat([scale.unsqueeze(-1), one_hot], dim=-1)  # (N, 5)


# --------------------------------------------------------------- point_cloud (512, 3)


def point_cloud_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env local-frame unit-scale point cloud, indexed by active geom."""
    return env.active_point_cloud()  # (N, 512, 3)
