"""Custom action term: absolute joint position relative to a fixed nominal grasp pose.

Semantics:
    target_joint = clamp(q_nominal + action * scale, soft_limits)

- action = 0   -> q_nominal (hand in a light-close grasp pose)
- action = +1  -> q_nominal + scale
- action = -1  -> q_nominal - scale

This replaces the standard JointPositionToLimitsAction, whose `action=0` maps
joints to the *midpoint* of the soft limits (which is far from a grasp pose
and causes the hand to open on the first step). With the grasp-relative
formulation, a freshly-initialized policy (outputs ~0) already holds a
reasonable grasp pose, avoiding the first ~100 iterations of wasted
samples during training.

Also applies an EMA on the processed targets (alpha configurable) to smooth
step-to-step changes, matching the paper's action smoothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


GRASP_NOMINAL_JOINT_POS: tuple[float, ...] = (
    -0.0117,  # index_0
    +0.0353,  # middle_0
    +0.1098,  # ring_0
    +1.1150,  # thumb_0
    +1.3358,  # index_1
    +1.2126,  # middle_1
    +1.3661,  # ring_1
    +1.0507,  # thumb_1
    +0.3071,  # index_2
    +0.1321,  # middle_2
    +0.2945,  # ring_2
    +0.9808,  # thumb_2
    +0.3159,  # index_3
    +0.1603,  # middle_3
    +0.3158,  # ring_3
    -0.0840,  # thumb_3
)
"""Nominal grasp pose: mean of all 58753 validated grasp joint angles.

This is a fixed, dataset-independent pose representing a light-close grasp.
With GraspRelativeAction, `action=0` maps to this pose instead of the joint
midpoint (which would fully open the hand and drop the object).
"""


class GraspRelativeAction(ActionTerm):
    """Absolute joint position action defined relative to a fixed nominal pose."""

    cfg: GraspRelativeActionCfg
    _asset: Articulation

    def __init__(self, cfg: GraspRelativeActionCfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        assert len(cfg.nominal_joint_pos) == self._num_joints, (
            f"nominal_joint_pos has {len(cfg.nominal_joint_pos)} entries, "
            f"expected {self._num_joints}"
        )
        self._nominal = torch.tensor(
            cfg.nominal_joint_pos, device=self.device, dtype=torch.float32
        )  # (num_joints,)
        self._scale = float(cfg.scale)
        self._alpha = float(cfg.alpha)

        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = self._nominal.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._prev_applied = self._processed_actions.clone()

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        clipped = actions.clamp(-1.0, 1.0)
        target = self._nominal.unsqueeze(0) + clipped * self._scale
        # EMA smoothing
        ema = self._alpha * target + (1.0 - self._alpha) * self._prev_applied
        # clamp to soft joint limits
        low = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        high = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        self._processed_actions[:] = torch.clamp(ema, low, high)
        self._prev_applied[:] = self._processed_actions

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        # reset EMA history to current joint position so the first step
        # starts from the real grasp pose (not a stale or zero value)
        self._prev_applied[env_ids] = self._asset.data.joint_pos[env_ids, self._joint_ids]


@configclass
class GraspRelativeActionCfg(ActionTermCfg):
    """Config for GraspRelativeAction."""

    class_type: type = GraspRelativeAction
    joint_names: list[str] = [".*"]
    nominal_joint_pos: list[float] = list(GRASP_NOMINAL_JOINT_POS)
    scale: float = 0.175
    alpha: float = 0.8
