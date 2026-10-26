"""DexReMoE actor-critic built on top of RSL-RL's ActorCritic.

The RSL-RL `OnPolicyRunner` hands the policy a single flat observation
tensor. We therefore flatten the four env obs groups into one
`OBS_LAYOUT.total`-dim tensor and slice it inside the actor / critic.
This keeps PPO / storage / optimizer / runner intact while still using
per-group normalization.

Layout (total 1660):
    policy_obs      [0:96]      3-step joint + 3-step action
    privileged      [96:119]    e_phys(19) + size_bucket_one_hot(4)
    point_cloud     [119:1655]  512 * 3, unit-scale local frame
    size_encoding   [1655:1660] [scale, b0..b3] (gate-only; unused here)

Normalization:
    policy_obs(96)            -> RunningMeanStd (inside network)
    privileged continuous(11) -> RunningMeanStd (inside network)
    everything else           -> passthrough
    f_t, z_t                  -> LayerNorm (inside PointNet / MuE)

actor and critic each own an independent PointNet + MuE. This mirrors the
paper (each has its own z_t) and avoids gradient interference between the
two heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic

from .mu_e import MuE
from .pointnet import PointNet
from .running_mean_std import RunningMeanStd


@dataclass(frozen=True)
class ObsLayout:
    policy_start: int = 0
    policy_dim: int = 96
    priv_start: int = 96
    priv_dim: int = 23
    point_start: int = 119
    point_points: int = 256
    point_coords: int = 3
    size_start: int = 1655
    size_dim: int = 5

    @property
    def point_flat_dim(self) -> int:
        return self.point_points * self.point_coords

    @property
    def total(self) -> int:
        return self.policy_dim + self.priv_dim + self.point_flat_dim + self.size_dim


OBS_LAYOUT = ObsLayout()

# Indices of continuous privileged features (excludes friction/com which are
# constant, quat_diff which is a unit quaternion, and the size one-hot).
PRIV_CONTINUOUS_INDICES: tuple[int, ...] = (0, 2, 6, 7, 8, 13, 14, 15, 16, 17, 18)


class _PrivNormalizer(nn.Module):
    """Normalize only the continuous privileged features, pass the rest through."""

    def __init__(self, dim: int, continuous_idx: tuple[int, ...]):
        super().__init__()
        self.register_buffer("idx", torch.tensor(continuous_idx, dtype=torch.long))
        self.norm = RunningMeanStd(len(continuous_idx))

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        cont = priv[:, self.idx]
        cont = self.norm(cont)
        out = priv.clone()
        out[:, self.idx] = cont
        return out


class _ObsEncoder(nn.Module):
    """Slice the flat obs, normalize per-group, fuse into z_t, and concat
    with the normalized policy observation.

    Output: (N, policy_dim + z_dim) — the shared trunk for actor / critic.
    """

    def __init__(self, policy_dim: int, priv_dim: int, z_dim: int = 66):
        super().__init__()
        self.policy_norm = RunningMeanStd(policy_dim)
        self.priv_norm = _PrivNormalizer(priv_dim, PRIV_CONTINUOUS_INDICES)
        self.pointnet = PointNet(out_dim=32)
        self.mu_e = MuE(in_dim=priv_dim + 32, hidden=128, out_dim=z_dim)
        self.trunk_dim = policy_dim + z_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        layout = OBS_LAYOUT
        policy_obs = obs[:, layout.policy_start : layout.policy_start + layout.policy_dim]
        priv = obs[:, layout.priv_start : layout.priv_start + layout.priv_dim]
        pc_flat = obs[:, layout.point_start : layout.point_start + layout.point_flat_dim]
        point_cloud = pc_flat.view(-1, layout.point_points, layout.point_coords)

        policy_obs = self.policy_norm(policy_obs)
        priv = self.priv_norm(priv)
        f_t = self.pointnet(point_cloud)          # (N, 32), LayerNorm-ed
        e_t = torch.cat([priv, f_t], dim=-1)       # (N, 55)
        z_t = self.mu_e(e_t)                       # (N, 66), LayerNorm-ed

        return torch.cat([policy_obs, z_t], dim=-1)  # (N, trunk_dim)


class DexReMoEActor(nn.Module):
    """Encoder + actor MLP."""

    def __init__(self, num_actions: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.encoder = _ObsEncoder(policy_dim=OBS_LAYOUT.policy_dim, priv_dim=OBS_LAYOUT.priv_dim)

        layers: list[nn.Module] = []
        c_in = self.encoder.trunk_dim
        for h in hidden_dims:
            layers.append(nn.Linear(c_in, h))
            layers.append(nn.ELU(inplace=True))
            c_in = h
        out = nn.Linear(c_in, num_actions)
        # Small init on the output layer -> initial action near 0, which maps
        # to q_nominal (light-close grasp) under GraspRelativeAction.
        nn.init.orthogonal_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs))


class DexReMoECritic(nn.Module):
    """Encoder + critic MLP. Independent from the actor's encoder."""

    def __init__(self, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.encoder = _ObsEncoder(policy_dim=OBS_LAYOUT.policy_dim, priv_dim=OBS_LAYOUT.priv_dim)

        layers: list[nn.Module] = []
        c_in = self.encoder.trunk_dim
        for h in hidden_dims:
            layers.append(nn.Linear(c_in, h))
            layers.append(nn.ELU(inplace=True))
            c_in = h
        layers.append(nn.Linear(c_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs))


class DexReMoEActorCritic(ActorCritic):
    """RSL-RL ActorCritic whose actor / critic are DexReMoE modules.

    Parent class already implements act / act_inference / evaluate /
    update_distribution / get_actions_log_prob using self.actor and
    self.critic, plus the action noise std. We only replace the two MLPs.

    The parent class stores num_actor_obs / num_critic_obs; we must pass
    OBS_LAYOUT.total so the distribution, storage, and optimizer see the
    right sizes.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if num_actor_obs != OBS_LAYOUT.total or num_critic_obs != OBS_LAYOUT.total:
            raise ValueError(
                f"num_actor_obs / num_critic_obs must equal OBS_LAYOUT.total={OBS_LAYOUT.total}; "
                f"got {num_actor_obs} / {num_critic_obs}."
            )
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        # Replace the placeholder MLPs created by the parent with the real ones.
        self.actor = DexReMoEActor(num_actions=num_actions, hidden_dims=tuple(actor_hidden_dims))
        self.critic = DexReMoECritic(hidden_dims=tuple(critic_hidden_dims))

    # ------------------------------------------------------------ exports

    def export_encoders(self) -> dict[str, dict[str, torch.Tensor]]:
        """Deep-copied state dicts for the encoder halves.

        `state_dict()` returns references into the live parameters, so a
        caller that keeps the result while the model keeps training would
        end up with a polluted snapshot. Clone every tensor.
        """
        def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
            return {k: v.detach().clone() for k, v in module.state_dict().items()}

        return {
            "actor_pointnet": _snapshot(self.actor.encoder.pointnet),
            "actor_mu_e": _snapshot(self.actor.encoder.mu_e),
            "critic_pointnet": _snapshot(self.critic.encoder.pointnet),
            "critic_mu_e": _snapshot(self.critic.encoder.mu_e),
        }

    def load_encoders(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Load encoder halves (e.g. from a base-policy checkpoint)."""
        self.actor.encoder.pointnet.load_state_dict(state["actor_pointnet"])
        self.actor.encoder.mu_e.load_state_dict(state["actor_mu_e"])
        self.critic.encoder.pointnet.load_state_dict(state["critic_pointnet"])
        self.critic.encoder.mu_e.load_state_dict(state["critic_mu_e"])

    def freeze_encoders(self, freeze_actor: bool = True, freeze_critic: bool = True) -> None:
        """Freeze encoder parameters (used in expert / gate training stages)."""
        def _freeze(mod: nn.Module) -> None:
            for p in mod.parameters():
                p.requires_grad = False

        if freeze_actor:
            _freeze(self.actor.encoder.pointnet)
            _freeze(self.actor.encoder.mu_e)
        if freeze_critic:
            _freeze(self.critic.encoder.pointnet)
            _freeze(self.critic.encoder.mu_e)
