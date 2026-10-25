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


def privileged_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """e_phys (19) + one_hot shape (3)."""
    obj = env.active_root_state_w()
    cmd: DexReMoECommand = env.command_manager.get_term("object_pose")

    # e_phys
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

    # one-hot by geom_idx
    one_hot = torch.zeros(env.num_envs, 3, device=env.device)
    one_hot.scatter_(1, env.active_geom_idx.unsqueeze(-1), 1.0)

    return torch.cat([e_phys, one_hot], dim=-1)  # (N, 22)


# --------------------------------------------------------------- size_encoding (5)


def size_encoding_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[scale, b0?, b1?, b2?, b3?] with b0 lower-open, b3 upper-open."""
    scale = env.active_scale()
    b0 = (scale <= 0.74).float()
    b1 = ((scale > 0.74) & (scale <= 0.78)).float()
    b2 = ((scale > 0.78) & (scale <= 0.82)).float()
    b3 = (scale > 0.82).float()
    return torch.stack([scale, b0, b1, b2, b3], dim=-1)  # (N, 5)


# --------------------------------------------------------------- point_cloud (512, 3)


def point_cloud_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env local-frame unit-scale point cloud, indexed by active geom."""
    return env.active_point_cloud()  # (N, 512, 3)
