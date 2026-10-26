"""DexReMoE reorientation env (manager-based).

Holds the 33-prim bookkeeping (`prims`, `active_prim_idx`, park offsets),
the point-cloud cache (3 unit-scale clouds indexed by geom), and the
`data_bucket` / `eval_mode` switches consumed by the reset event.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..cache import GraspCache


class DexReMoEReorientEnv(ManagerBasedRLEnv):
    """Allegro reorientation env with 33 prims (11 scales x 3 shapes)."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # create the cache before super().__init__ so the reset event sees it
        if cfg.events.reset.params.get("cache") is None:
            cfg.events.reset.params["cache"] = GraspCache(cfg.cache_root, cfg.cache_split)
        # pre-seed buffers so obs shape probing inside load_managers() succeeds
        self.prims: list = []
        self.active_prim_idx = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        self.object_pos_init = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
        self.object_size_m = torch.full((cfg.scene.num_envs,), 0.064, device=cfg.sim.device)
        self._prim_park_offset = torch.zeros(0, 3, device=cfg.sim.device)
        self._point_clouds = None
        self.eval_offset = 0  # for eval_mode: env i uses eval pool row (offset + i) % n_eval
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self) -> None:
        """Build prims/point clouds before managers (obs shape probing) run."""
        self._build_prims()
        self._build_point_clouds()
        super().load_managers()

    # ---------------------------------------------------------------- setup

    def _build_prims(self) -> None:
        n_scales = len(self.cfg.scale_names)
        n_geoms = len(self.cfg.geom_names)
        self.prims = []
        for s_idx in range(n_scales):
            for g_idx in range(n_geoms):
                name = f"obj_{s_idx}_{g_idx}"
                self.prims.append(self.scene[name])
        assert len(self.prims) == n_scales * n_geoms

        # active prim index (0..32); preserve any pre-seeded value
        if self.active_prim_idx.shape[0] != self.num_envs:
            self.active_prim_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # initial state buffers
        if self.object_pos_init.shape[0] != self.num_envs:
            self.object_pos_init = torch.zeros(self.num_envs, 3, device=self.device)
            self.object_size_m = torch.full((self.num_envs,), 0.064, device=self.device)

        # park offsets for the 32 inactive prims
        self._prim_park_offset = torch.zeros(len(self.prims), 3, device=self.device)
        for i in range(len(self.prims)):
            s_idx = i // n_geoms
            scale = self.cfg.scale_values[self.cfg.scale_names[s_idx]]
            size = self.cfg.hora_base_size_m * scale
            row = i // 6 - 2.5
            col = i % 6 - 2.5
            self._prim_park_offset[i] = torch.tensor(
                [col * 0.2, row * 0.2, size / 2.0], device=self.device
            )

    def _build_point_clouds(self) -> None:
        """Precompute 3 unit-scale point clouds (cuboid / cylinder / sphere)."""
        rng = np.random.default_rng(0)
        n = self.cfg.point_cloud_points
        clouds = []

        # cuboid: sample on surface of unit cube [-0.5, 0.5]^3
        def sample_cuboid():
            pts = rng.uniform(-0.5, 0.5, size=(n, 3))
            axis = rng.integers(0, 3, size=n)
            side = rng.choice([-0.5, 0.5], size=n)
            pts[np.arange(n), axis] = side
            return pts.astype(np.float32)

        # cylinder: radius 0.5, height 1.0, axis = Z (matches CylinderCfg default).
        # Full surface: side wall + both end caps.
        def sample_cylinder():
            n_side = int(n * 0.6)
            n_cap_top = (n - n_side) // 2
            n_cap_bot = n - n_side - n_cap_top

            # side wall: r = 0.5, z uniform
            theta = rng.uniform(0, 2 * np.pi, size=n_side)
            z = rng.uniform(-0.5, 0.5, size=n_side)
            side = np.stack(
                [0.5 * np.cos(theta), 0.5 * np.sin(theta), z], axis=-1
            )

            # end caps: disk sampled area-uniformly (r = 0.5 * sqrt(u))
            def disk(z_val: float, count: int) -> np.ndarray:
                u = rng.uniform(0.0, 1.0, size=count)
                theta = rng.uniform(0, 2 * np.pi, size=count)
                r = 0.5 * np.sqrt(u)
                return np.stack(
                    [r * np.cos(theta), r * np.sin(theta), np.full(count, z_val)],
                    axis=-1,
                )

            top = disk(+0.5, n_cap_top)
            bot = disk(-0.5, n_cap_bot)
            return np.concatenate([side, top, bot], axis=0).astype(np.float32)

        # sphere: radius 0.5, uniform on surface
        def sample_sphere():
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
            return (v * 0.5).astype(np.float32)

        for fn in (sample_cuboid, sample_cylinder, sample_sphere):
            clouds.append(fn())

        self._point_clouds = torch.from_numpy(np.stack(clouds, axis=0)).to(self.device)  # (3, n, 3)

    # ------------------------------------------------------------ accessors

    @property
    def active_geom_idx(self) -> torch.Tensor:
        return (self.active_prim_idx % 3).long()

    @property
    def active_scale_idx(self) -> torch.Tensor:
        return (self.active_prim_idx // 3).long()

    def active_scale(self) -> torch.Tensor:
        vals = torch.tensor(
            [self.cfg.scale_values[n] for n in self.cfg.scale_names], device=self.device
        )
        return vals[self.active_scale_idx]

    def active_mass(self) -> torch.Tensor:
        # mass is stored per prim; gather from the active prim's physx view
        out = torch.zeros(self.num_envs, device=self.device)
        if not self.prims:
            return out
        for i, prim in enumerate(self.prims):
            mask = (self.active_prim_idx == i)
            if mask.any():
                m = prim.root_physx_view.get_masses().cpu().squeeze(-1)
                out[mask] = m[mask.cpu()].to(self.device)
        return out

    def active_root_state_w(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Gather pos/quat/lin_vel/ang_vel of the active prim (per env, or per env_ids)."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
        n = int(ids.numel())
        if n == 0:
            return {k: torch.zeros((0, d), device=self.device) for k, d in (("pos", 3), ("quat", 4), ("lin_vel", 3), ("ang_vel", 3))}
        if not self.prims:
            # shape probing during load_managers(); return zeros of the right shape
            return {
                "pos": torch.zeros(n, 3, device=self.device),
                "quat": torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4).contiguous(),
                "lin_vel": torch.zeros(n, 3, device=self.device),
                "ang_vel": torch.zeros(n, 3, device=self.device),
            }

        prim_idx = self.active_prim_idx[ids]

        pos = torch.stack([p.data.root_pos_w for p in self.prims], dim=1)[ids]
        quat = torch.stack([p.data.root_quat_w for p in self.prims], dim=1)[ids]
        lin = torch.stack([p.data.root_lin_vel_w for p in self.prims], dim=1)[ids]
        ang = torch.stack([p.data.root_ang_vel_w for p in self.prims], dim=1)[ids]

        # gather per env
        pos = pos.gather(1, prim_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        quat = quat.gather(1, prim_idx.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
        lin = lin.gather(1, prim_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        ang = ang.gather(1, prim_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

        return {"pos": pos, "quat": quat, "lin_vel": lin, "ang_vel": ang}

    def active_point_cloud(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """(N, n_points, 3) unit-scale point cloud for the active geom."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
        if self._point_clouds is None:
            return torch.zeros(ids.numel(), self.cfg.point_cloud_points, 3, device=self.device)
        geom = self.active_geom_idx[ids]
        return self._point_clouds[geom]  # (K, n_points, 3)

    # -------------------------------------------------------------- switches

    @property
    def eval_mode(self) -> bool:
        return bool(getattr(self, "_eval_mode", False))

    @eval_mode.setter
    def eval_mode(self, value: bool) -> None:
        self._eval_mode = bool(value)
        # force a hard reset of all envs so the next episode uses the new mode
        self.reset(env_ids=torch.arange(self.num_envs, device=self.device))
