"""Reset event: pull a grab from the validated cache and inject it.

Responsibilities:
  - Activate the prim that matches (scale_idx, geom_idx) of the sampled grab;
    park all other 32 prims on the ground staging grid.
  - Write joint state + position target (target is required, otherwise PD
    pulls fingers back to the default open pose).
  - Write object root state (pos must be `pos + env_origins`).
  - Set the mass on the active prim only (parked prims keep spawn default).
  - Record `object_pos_init` and `object_size_m` for termination checks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase


class ResetFromGraspCache(ManagerTermBase):
    """ManagerTermBase that owns the cache-driven reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cache = cfg.params["cache"]
        self._sample_rng = np.random.default_rng(cfg.params.get("seed", 0))
        self._eval_rng = torch.Generator(device="cpu")
        self._eval_rng.manual_seed(cfg.params.get("eval_seed", 0))

    def __call__(self, env, env_ids: Sequence[int], cache=None, seed=0, eval_seed=0):
        # `cache`, `seed`, `eval_seed` are declared for EventManager signature
        # validation; values are captured in __init__ already.
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        K = int(env_ids.numel())
        if K == 0:
            return

        # --- 1. sample ---
        if env.eval_mode:
            data = self._cache.sample_eval(
                n_per_file=env.cfg.eval_n_per_file, seed=env.cfg.eval_seed
            )
            n_eval = int(data["joint"].shape[0])
            idx = torch.randint(0, n_eval, (K,), generator=self._eval_rng).to(data["joint"].device)
            data = {
                k: (v[idx] if isinstance(v, torch.Tensor) and v.shape[0] == n_eval else v)
                for k, v in data.items()
            }
        else:
            data = self._cache.sample(K, rng=self._sample_rng, bucket=env.cfg.data_bucket)

        joint = data["joint"]
        pos = data["pos"]
        quat = data["quat"]
        mass = data["mass"]
        geom_idx = data["geom_idx"]
        scale_idx = data["scale_idx"]
        size_m = data["size_m"]

        prim_idx = scale_idx * 3 + geom_idx  # (K,) 0..32

        # --- 2. joint state (both state and target) ---
        robot = env.scene["robot"]
        robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=env_ids)
        robot.set_joint_position_target(joint, env_ids=env_ids)

        # --- 3. activate the right prim, park the rest ---
        env.active_prim_idx[env_ids] = prim_idx

        world_pos = pos + env.scene.env_origins[env_ids]
        root = torch.zeros(K, 13, device=env.device)
        root[:, 0:3] = world_pos
        root[:, 3:7] = quat

        for prim_id, prim in enumerate(env.prims):
            active_mask = (prim_idx == prim_id)
            if active_mask.any():
                local_ids = env_ids[active_mask]
                prim.write_root_state_to_sim(root[active_mask], env_ids=local_ids)
                masses_full = prim.root_physx_view.get_masses().clone().cpu()
                ids_cpu = local_ids.detach().cpu().int().contiguous()
                masses_full[ids_cpu] = mass[active_mask].cpu().unsqueeze(-1)
                prim.root_physx_view.set_masses(masses_full, ids_cpu)

            park_mask = (prim_idx != prim_id)
            if park_mask.any():
                local_ids = env_ids[park_mask]
                park = torch.zeros(int(park_mask.sum()), 13, device=env.device)
                park[:, 0:3] = (
                    env.scene.env_origins[local_ids] + env._prim_park_offset[prim_id]
                )
                prim.write_root_state_to_sim(park, env_ids=local_ids)

        # --- 4. bookkeeping ---
        env.object_pos_init[env_ids] = world_pos
        env.object_size_m[env_ids] = size_m

        # --- 5. flush (no physics step; matches validation protocol) ---
        env.scene.write_data_to_sim()
