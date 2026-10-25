# DexReMoE Base-Policy Env Design

## 1. Package

`DexReMoERebuild/source/dexremoe_tasks/` (editable install). Env-only; all networks live in the trainer.

- `envs/reorient_env_cfg.py` — `DexReMoEReorientEnvCfg`
- `envs/reorient_env.py` — `DexReMoEReorientEnv`
- `envs/mdp/` — commands, events, observations, rewards, terminations
- `cache/grasp_cache.py` — `GraspCache`
- `assets/allegro_cfg.py` — `ALLEGRO_DEXREMOE_CFG` (fix_root_link, Kp=3.0/Kd=1.0, effort=0.5, hand root z=0.6)

## 2. Observation groups

| group | shape | content | used by |
|---|---|---|---|
| `policy` | (N, 96) | 3-step joint + 3-step last action | policy |
| `privileged` | (N, 22) | e_phys 19 + shape one-hot 3 | policy |
| `point_cloud` | (N, 512, 3) | unit-scale local-frame cloud, 3 precomputed groups by geom | PointNet++ (trainer) |
| `size_encoding` | (N, 5) | [scale, b0?, b1?, b2?, b3?] | gate (trainer) |

e_phys = [mass, friction, scale, com(3), pos_diff(3), quat_diff(4), lin_vel(3), ang_vel(3)].

- com: constant 0.
- friction: constant 1.0 (global physics material).
- pos_diff / lin_vel / ang_vel: world-frame differences/velocities; env_origin cancels, so numerically equal to env-frame values.
- quat_diff = quat_obj * conj(quat_goal), w >= 0.

size_encoding buckets: b0 = scale <= 0.74, b1 = 0.74 < scale <= 0.78, b2 = 0.78 < scale <= 0.82, b3 = scale > 0.82. OOD (s066 / s09) falls into b0 / b3 naturally.

## 3. Action

mdp.EMAJointPositionToLimitsActionCfg(alpha=0.8, rescale_to_limits=True). Absolute joint position; output clipped to [-1, 1] then rescaled to soft joint limits.

## 4. Success / failure

DexReMoECommand (independent rewrite, inherits CommandTerm):

- Goal orientation: uniform SO(3), with min_initial_angle = pi/2 guard.
- Goal position: pinned at object position at reset; not resampled on success.
- Sustained-hold success: 4 conditions (tau_theta=0.1, tau_q=10.0, tau_v=0.04, tau_w=0.5) held for 5 consecutive control steps.
- On success: consecutive_success += 1, just_succeeded = True for 1 step, resample goal, no termination.
- Failure reset: dtheta > 2 * tau_theta while consecutive_success > 0 leads to consecutive_success = 0; no termination.
- Exposes just_succeeded (consumed by success_bonus reward) and get_max_consecutive_success().

## 5. Termination

| term | rule |
|---|---|
| time_out | 600 steps (10s @ 60Hz) |
| object_dropped | obj_z < pos_init_z - 2 * size |
| object_far | ||obj_pos - pos_init|| > 3 * size |
| max_success | max_consecutive_success >= 25 |

## 6. Reward

| term | weight | source |
|---|---|---|
| success | +800 | one-shot on just_succeeded |
| orientation | +1.0 | 1 / (dtheta + 0.1) |
| pos_penalty | -10.0 | ||pos_diff|| |
| joint_vel | -0.0002 | sum_i max(abs(w_i) - 6, 0) |
| action_l2 | -0.0001 | ||a||^2 |
| action_rate | -0.01 | ||a_t - a_{t-1}||^2 |

No reward normalization in env; trainer decides.

## 7. Reset (cache -> sim)

ResetFromGraspCache.__call__:

1. GraspCache.sample(K, bucket=cfg.data_bucket) or sample_eval (if env.eval_mode).
2. robot.write_joint_state_to_sim(joint, zeros) plus robot.set_joint_position_target(joint). Both required.
3. env.active_prim_idx[ids] = scale_idx * 3 + geom_idx.
   - Active prim: write_root_state_to_sim(pos + env_origins, quat) and set_masses (full array).
   - All 32 other prims: parked at env_origins + _prim_park_offset, ground grid, 0.2 m spacing.
4. env.object_pos_init = world_pos; env.object_size_m = size.
5. scene.write_data_to_sim(). No physics step.

## 8. Scene

- 33 prims: 11 scales (s07..s086, s066, s09) x 3 geoms (cuboid / cylinder / sphere).
- env_spacing = 3.0, hand root z = 0.6.
- Global physics material: static / dynamic friction 1.0.
- sim.dt = 1/120, decimation = 2 (60 Hz control), 10s episode = 600 steps.
- num_envs = 128.

## 9. Three-stage switch

Env exposes two flags:

- cfg.data_bucket: None (stratified 1/3 across small/medium/large; for base and gate) or one of "small" / "medium" / "large" (expert training).
- env.eval_mode: True switches sampling to cache.sample_eval() and forces a hard reset.

No phase semantics inside the env. Each training stage is a separate script that sets cfg.data_bucket once.

## 10. Cache files

assets/grasp_cache_isaac_validated/<split>/grasp_<scale>_<shape>.npz with fields: joint_isaac, obj_pos_isaac, obj_quat_wxyz, mass, friction, shape_id, scale, com_offset, source_row.

- The file suffix (cuboid / cylinder / sphere) is the authoritative geometry; shape_id is a random preprocess label and MUST NOT be used to pick geometry or enter observations.
- obj_pos_isaac is in single-env world coords; reset adds env.scene.env_origins.
- friction and com_offset are ignored (global friction 1.0, com = 0).

## 11. PointNet++ / mu_e / gate (trainer-side)

Env emits raw point_cloud and size_encoding. Trainer composes:

    f_t     = PointNet++(obs["point_cloud"])      # (N, 32), unit-scale local cloud
    gate_in = cat([f_t, obs["size_encoding"]], -1) # (N, 37)
    e_t     = cat([obs["privileged"], f_t], -1)    # (N, 54)
    z_t     = mu_e(e_t)                             # (N, 66)
    action  = policy(cat([obs["policy"], z_t], -1))

obs["policy"] uses empirical normalization. privileged, size_encoding, point_cloud are not normalized.
