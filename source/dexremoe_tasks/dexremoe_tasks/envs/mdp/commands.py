"""DexReMoE in-hand reorientation command.

Independent rewrite: inherits only from `isaaclab.managers.CommandTerm`,
not from the upstream inhand orientation command. Implements:

  - Absolute orientation goal, uniformly sampled from SO(3), with the
    constraint that the initial rotation error to the current object pose
    is >= `min_initial_angle` (default 90 deg).
  - Position goal fixed at the object position captured at reset.
  - Sustained-hold success: 4 conditions (orientation error, joint velocity,
    object linear velocity, object angular velocity) all satisfied for N
    consecutive control steps.
  - On success: consecutive_success += 1, resample goal, set just_succeeded
    for this step; episode is NOT terminated.
  - If a success streak exists and dtheta grows past 2*tau_theta: reset
    consecutive_success to 0 (no termination).
  - `just_succeeded` is surfaced via a property, consumed by the reward term.

Timing note: `_update_metrics` runs inside `command_manager.compute()` which
fires AFTER `reward_manager.compute()` in `ManagerBasedRLEnv.step()`. So the
reward term sees a one-step delayed success flag. This is acceptable for a
sparse +800 event and avoids double-computing the hold counter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


def _sample_uniform_quat(n: int, device: str | torch.device) -> torch.Tensor:
    """Sample n quaternions uniformly from SO(3), returned in (w, x, y, z) order."""
    u = torch.rand(n, 3, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    w = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    x = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    y = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    z = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    return torch.stack([w, x, y, z], dim=-1)


class DexReMoECommand(CommandTerm):
    """SO(3) goal command + sustained-hold success tracker."""

    cfg: DexReMoECommandCfg

    def __init__(self, cfg: DexReMoECommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.robot_name]

        # absolute goal tensors
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0

        # internal state
        self._hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._just_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_consec = torch.zeros(self.num_envs, device=self.device)

        # metrics (surfaced to eval)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_counter"] = torch.zeros(self.num_envs, device=self.device)

    # ---------------------------------------------------------------- props

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.pos_command_w, self.quat_command_w], dim=-1)  # (N, 7)

    @property
    def just_succeeded(self) -> torch.Tensor:
        return self._just_succeeded

    def get_max_consecutive_success(self) -> torch.Tensor:
        return self._max_consec

    # ---------------------------------------------------------------- reset

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        info = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._hold[ids] = 0
        self._just_succeeded[ids] = False
        self._max_consec[ids] = 0.0
        self.metrics["consecutive_success"][ids] = 0.0
        self.metrics["max_consecutive_success"][ids] = 0.0
        self.metrics["hold_counter"][ids] = 0.0
        return info

    # ------------------------------------------------------- resample goal

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device)
        n = int(ids.numel())
        if n == 0:
            return

        obj = self._env.active_root_state_w(ids)
        # position goal is pinned at reset; never resampled afterwards
        self.pos_command_w[ids] = obj["pos"]

        goal = self._sample_goal_with_guard(obj["quat"], n)
        self.quat_command_w[ids] = math_utils.quat_unique(goal) if self.cfg.make_quat_unique else goal

    def _resample_goal_only(self, ids: torch.Tensor) -> None:
        n = int(ids.numel())
        obj = self._env.active_root_state_w(ids)
        goal = self._sample_goal_with_guard(obj["quat"], n)
        self.quat_command_w[ids] = math_utils.quat_unique(goal) if self.cfg.make_quat_unique else goal

    def _sample_goal_with_guard(self, current_quat: torch.Tensor, n: int) -> torch.Tensor:
        goal = _sample_uniform_quat(n, device=self.device)
        for _ in range(10):
            dtheta = math_utils.quat_error_magnitude(current_quat, goal)
            bad = dtheta < self.cfg.min_initial_angle
            if not bad.any():
                break
            goal[bad] = _sample_uniform_quat(int(bad.sum()), device=self.device)
        return goal

    def _update_command(self) -> None:
        # Success-driven resample is handled in `_update_metrics`.
        pass

    # ----------------------------------------------------------- metrics

    def _update_metrics(self) -> None:
        obj = self._env.active_root_state_w()
        robot = self.robot

        dtheta = math_utils.quat_error_magnitude(obj["quat"], self.quat_command_w)
        cond_orient = dtheta < self.cfg.tau_theta
        cond_joint = (robot.data.joint_vel.abs() < self.cfg.tau_q).all(dim=-1)
        cond_linvel = obj["lin_vel"].norm(dim=-1) < self.cfg.tau_v
        cond_angvel = obj["ang_vel"].norm(dim=-1) < self.cfg.tau_w
        all_ok = cond_orient & cond_joint & cond_linvel & cond_angvel

        self._hold = torch.where(all_ok, self._hold + 1, torch.zeros_like(self._hold))

        just_ok = self._hold >= self.cfg.sustained_hold_steps
        if just_ok.any():
            ids = just_ok.nonzero(as_tuple=False).squeeze(-1)
            self.metrics["consecutive_success"][ids] += 1.0
            self._max_consec[ids] = torch.maximum(
                self._max_consec[ids], self.metrics["consecutive_success"][ids]
            )
            self._just_succeeded[ids] = True
            # zero for non-success envs
            self._just_succeeded &= just_ok
            self._hold[ids] = 0
            self._resample_goal_only(ids)
        else:
            self._just_succeeded[:] = False

        # failure reset (no termination)
        far = (dtheta > 2.0 * self.cfg.tau_theta) & (self.metrics["consecutive_success"] > 0)
        if far.any():
            ids = far.nonzero(as_tuple=False).squeeze(-1)
            self.metrics["consecutive_success"][ids] = 0.0

        self.metrics["max_consecutive_success"] = self._max_consec.clone()
        self.metrics["hold_counter"] = self._hold.float()
        # Diagnostic: orientation error in degrees. Direct read is far easier
        # than back-solving from Episode_Reward/orientation.
        self.metrics["dtheta_deg"] = dtheta * 180.0 / math.pi


@configclass
class DexReMoECommandCfg(CommandTermCfg):
    """Config for :class:`DexReMoECommand`."""

    class_type: type = DexReMoECommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    robot_name: str = "robot"

    # success thresholds (rad, rad/s, m/s, rad/s)
    # tau_theta relaxed from 0.1 to 0.15 so a fresh policy (avg err ~87 deg)
    # can occasionally trigger success and get the +800 signal.
    tau_theta: float = 0.15
    tau_q: float = 10.0
    tau_v: float = 0.04
    tau_w: float = 0.5

    # 5 -> 3: the policy is still jittery early on; 5 consecutive steps is
    # essentially unreachable, so success never fires.
    sustained_hold_steps: int = 3
    min_initial_angle: float = math.pi / 2.0
    make_quat_unique: bool = True
