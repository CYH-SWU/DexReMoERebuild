# DexReMoE Base-Policy Design

> Scope: base policy only. MoE (expert / gate) interfaces are stubbed where they matter but not trained.

## 1. Packages

### 1.1 Env — `source/dexremoe_tasks/` (editable install)

Env-only. No network layers. Produces raw observation groups.

- `envs/reorient_env_cfg.py` — `DexReMoEReorientEnvCfg`
- `envs/reorient_env.py` — `DexReMoEReorientEnv`
- `envs/mdp/` — actions, commands, events, observations, rewards, terminations
- `cache/grasp_cache.py` — `GraspCache`
- `assets/allegro_cfg.py` — `ALLEGRO_DEXREMOE_CFG`

### 1.2 Networks — `source/dexremoe_agents/` (editable install)

Training-side networks. Consumed by the trainer; not imported by the env.

- `networks/running_mean_std.py` — `RunningMeanStd`
- `networks/pointnet.py` — `PointNet` (vanilla, not PointNet++)
- `networks/mu_e.py` — `MuE`
- `networks/actor_critic.py` — `DexReMoEActorCritic` (inherits RSL-RL `ActorCritic`)

## 2. Observation groups

| group | shape | content | used by |
|---|---|---|---|
| `policy` | (N, 96) | 3-step joint + 3-step last action | policy |
| `privileged` | (N, 23) | e_phys 19 + size bucket one-hot 4 | policy |
| `point_cloud` | (N, 256, 3) | unit-scale local-frame cloud, 3 precomputed groups by geom | PointNet (trainer) |
| `size_encoding` | (N, 5) | `[scale, b0?, b1?, b2?, b3?]` | gate (MoE only) |

Point clouds are generated at env init with `np.random.default_rng(0)`, kept on GPU, and never written to disk. The seed is fixed, so every run sees the same 3 clouds. Coverage: cuboid = 6 faces; sphere = full sphere; cylinder = side wall (60%) + top cap (20%) + bottom cap (20%), matching the `CylinderCfg(axis="Z")` physics geometry.

`e_phys = [mass, friction, scale, com(3), pos_diff(3), quat_diff(4), lin_vel(3), ang_vel(3)]`.

- `com`: constant 0.
- `friction`: constant 1.0 in the observation (placeholder privileged field; the actual physics material used by the env is 0.5/0.5, matching the cache validation).
- `pos_diff` / `lin_vel` / `ang_vel`: world-frame differences/velocities; env_origin cancels, so numerically equal to env-frame values.
- `quat_diff = quat_obj * conj(quat_goal)`, w >= 0.

Shape is **not** encoded in `privileged`: the PointNet is the sole source of shape information. `privileged[19:23]` is the 4-dim size-bucket one-hot, duplicating the `scale` scalar at `e_phys[2]` on purpose (continuous + discrete cue).

Size bucket boundaries (with float32 eps 1e-4):

| bucket | range |
|---|---|
| b0 | scale <= 0.74 |
| b1 | 0.74 < scale <= 0.78 |
| b2 | 0.78 < scale <= 0.82 |
| b3 | scale > 0.82 |

OOD (s066 / s09) falls into b0 / b3 naturally. The same helper `_size_bucket_one_hot` feeds both `privileged` and `size_encoding` so float32 boundary cases stay consistent.

## 3. Action

`GraspRelativeActionCfg(joint_names=[".*"], scale=0.175, alpha=0.8, nominal_joint_pos=GRASP_NOMINAL_JOINT_POS)`.

    target_joint = clamp(q_nominal + action * scale, soft_limits)
    target_joint = alpha * target_joint + (1 - alpha) * prev_applied   # EMA

- action = 0   -> q_nominal (fixed light-close grasp, mean of 58,753 validated cache grasps)
- action = +1  -> q_nominal + 0.175
- action = -1  -> q_nominal - 0.175

`q_nominal` is a hard-coded constant independent of which cache row was sampled. Reset seeds the EMA history with the actual dataset grasp pose, so the first few steps smoothly converge from the sampled grasp to `q_nominal`.

Why not `JointPositionToLimitsAction`: its `action=0` maps joints to the **midpoint** of soft limits (open hand), causing every freshly-initialized policy to release the object on step 1 and wasting ~100 iterations of samples.

## 4. Success / failure

`DexReMoECommand` (independent rewrite, inherits `CommandTerm`):

- Goal orientation: uniform SO(3), with `min_initial_angle = pi/2` guard.
- Goal position: pinned at object position at reset; not resampled on success.
- Sustained-hold success: 4 conditions (`tau_theta=0.15`, `tau_q=10.0`, `tau_v=0.04`, `tau_w=0.5`) held for 3 consecutive control steps.
- On success: `consecutive_success += 1`, `just_succeeded = True` for 1 step, resample goal, **no termination**.
- Failure reset: `dtheta > 2 * tau_theta` while `consecutive_success > 0` leads to `consecutive_success = 0`; no termination.
- Exposes `just_succeeded` (consumed by `success_bonus` reward) and `get_max_consecutive_success()`.
- Metrics surfaced for logging: `consecutive_success`, `max_consecutive_success`, `hold_counter`, `dtheta_deg`.

> The 0.15 rad / 3-step thresholds are relaxed from the paper (0.1 rad / 5 steps). With 64 envs the policy stays around ~87 deg orientation error for the first few hundred iterations; at the paper thresholds `success` never fires, so the policy never receives the +800 signal and stalls in a "hold still, don't rotate" local optimum.

## 5. Termination

| term | rule |
|---|---|
| `time_out` | 600 steps (10 s @ 60 Hz) |
| `object_dropped` | `obj_z < pos_init_z - 2 * size` |
| `object_far` | `||obj_pos - pos_init|| > 3 * size` |
| `max_success` | `max_consecutive_success >= 25` |

## 6. Reward

| term | weight | source |
|---|---|---|
| `success` | +800 | one-shot on `just_succeeded` |
| `orientation` | +2.0 | `1 / (dtheta + 0.1)` |
| `pos_penalty` | -5.0 | `||pos_diff||` |
| `joint_vel` | -0.0002 | `sum_i max(abs(w_i) - 6, 0)` |
| `action_l2` | -0.0001 | `||a||^2` |
| `action_rate` | -0.005 | `||a_t - a_{t-1}||^2` |

No reward normalization inside env; the trainer decides.

> The weights differ from paper Table II (`c_rot=+1.0`, `c_dist=-10.0`, no `action_rate`). `orientation` is doubled, `pos_penalty` and `action_rate` are halved, because at 64 envs the per-step negative terms dominate the accumulated reward once episodes grow from ~50 to ~500 steps. See `docs/training.md` for the full reasoning.

## 7. Reset (cache -> sim)

`ResetFromGraspCache.__call__`:

1. `GraspCache.sample(K, bucket=cfg.data_bucket)` or `sample_eval` (if `env.eval_mode`).
2. `robot.write_joint_state_to_sim(joint, zeros)` **plus** `robot.set_joint_position_target(joint)`. Both required; otherwise PD pulls fingers back to the default open pose.
3. `env.active_prim_idx[ids] = scale_idx * 3 + geom_idx`.
   - Active prim: `write_root_state_to_sim(world_pos, quat)`, `set_masses` (full array + CPU int64 contiguous ids).
   - All 32 other prims: parked at `env_origins + _prim_park_offset`, 6x6 ground grid, 0.2 m spacing.
4. **Object z offset**: the cache stores object positions relative to the hand root at `cache_hand_z = 0.5` (the hand z used during validation). The reset shifts z by `hand_z_now - cache_hand_z` so the object always sits at the fingertip regardless of the current hand z.
5. `env.object_pos_init = world_pos`; `env.object_size_m = size`.
6. `scene.write_data_to_sim()`. No physics step.

## 8. Scene

- 33 prims: 11 scales (s07, s072, s074, s076, s078, s08, s082, s084, s086, s066, s09) x 3 geoms (cuboid / cylinder / sphere).
- `env_spacing = 3.0`, **hand root z = 1.0** (raised from 0.5 so dropped objects never reach the parked prims within a 60-step check).
- `cache_hand_z = 0.5` (constant used to shift object z; see section 7 step 4).
- Physics: `sim.dt = 1/60`, `decimation = 1` (60 Hz control), 10 s episode = 600 steps.
- `SimulationCfg` uses defaults (friction 0.5/0.5, bounce 0.5) to match `validate_grasp_stability.py`. Do **not** override `physics_material` or `physxCfg` unless the cache is also re-validated.
- `num_envs = 64` for base training (a 128-env run OOMs on the 8 GiB RTX 5060 when PointNet backward fires; 64 stays under 5.5 GiB).
- Allegro hand: `fix_root_link=True`, implicit actuator `Kp=3.0 / Kd=1.0`, `effort_limit_sim=0.5`.

## 9. Three-stage switch

Env exposes two flags:

- `cfg.data_bucket`: `None` (stratified 1/3 across small/medium/large; for base and gate) or one of `"small"` / `"medium"` / `"large"` (expert training).
- `env.eval_mode`: `True` switches sampling to `cache.sample_eval()` and forces a hard reset. Sampling in eval mode is a fixed mapping `env i -> eval_pool[(eval_offset + i) % n_eval]`, so two consecutive resets give identical rows. `eval_offset` lets an eval script sweep the whole pool in chunks.

No phase semantics inside the env; each training stage is a separate script that sets `cfg.data_bucket` once.

## 10. Cache files

`assets/grasp_cache_isaac_validated/<split>/grasp_<scale>_<shape>.npz` with fields: `joint_isaac`, `obj_pos_isaac`, `obj_quat_wxyz`, `mass`, `friction`, `shape_id`, `scale`, `com_offset`, `source_row`.

- The file suffix (`cuboid` / `cylinder` / `sphere`) is the authoritative geometry. `shape_id` is a random preprocess label and MUST NOT be used to pick geometry or enter observations.
- `obj_pos_isaac` is in single-env world coords; reset adds `env.scene.env_origins` and the z offset from section 7 step 4.
- `friction` and `com_offset` are ignored (global friction 0.5, com = 0).
- The cache was validated with the hand **free-floating** at z=0.5. Replaying it against a fixed hand at z=1.0 keeps ~75% of the entries within tolerance; the 25% drift is the accepted gap (see `.vscode/log.txt` section 9).

## 11. Network design (base policy, dexremoe_agents)

### 11.1 Flattened observation

RSL-RL only hands a single flat tensor to `act()`. `OBS_LAYOUT` in `actor_critic.py`:

| slice | dim | content |
|---|---|---|
| `[0:96]` | 96 | policy |
| `[96:119]` | 23 | privileged |
| `[119:887]` | 768 | point_cloud (256 x 3, flattened) |
| `[887:892]` | 5 | size_encoding (unused in base) |

Total `OBS_LAYOUT.total = 892`. The actor and critic slice and reshape this internally.

### 11.2 Normalization

| signal | normalizer | reason |
|---|---|---|
| `policy` (96) | `RunningMeanStd` | joint / action scales vary per joint |
| `privileged` continuous 11 dims | `RunningMeanStd` | mass, scale, pos_diff, lin_vel, ang_vel are continuous |
| `privileged` constant / bounded / discrete 12 dims | passthrough | friction & com are constants (var=0 -> NaN if normalized); quat_diff is a unit quaternion; one-hot is 0/1 |
| `point_cloud` (256x3) | passthrough | already unit-scale; per-axis mean/std would distort geometry |
| `size_encoding` (5) | passthrough | base ignores it |
| `f_t` (32) | `LayerNorm` | network output; distribution shifts during training |
| `z_t` (66) | `LayerNorm` | same |

`PRIV_CONTINUOUS_INDICES = (0, 2, 6, 7, 8, 13, 14, 15, 16, 17, 18)`.

RSL-RL's built-in `empirical_normalization` is **off**; all normalization lives inside `DexReMoEActorCritic` and its buffers travel with `state_dict()`.

### 11.3 Encoders

    f_t = PointNet(point_cloud)                         # (N, 32), LayerNorm-ed
    e_t = cat([privileged_norm(23), f_t(32)])           # (N, 55)
    z_t = MuE(e_t)                                      # (N, 66), LayerNorm-ed

- **PointNet**: `Conv1d(3->64->128->512)` + BN + ELU -> global max pool -> `Linear(512->512->256->32)` + ELU + `LayerNorm(32)`. Vanilla PointNet, not PointNet++, because the three geometries are convex primitives. The 1024-channel variant OOMs on this GPU; 512 channels keeps the backward activation under ~250 MiB per mini-batch.
- **MuE**: `Linear(55->128)` + ELU + `Linear(128->66)` + `LayerNorm(66)`.

### 11.4 Actor and critic

- Independent PointNet + MuE for each head (matches the paper; avoids gradient interference).
- `actor_in  = cat([policy_norm(96), z_t(66)]) = 162`
- `critic_in = cat([policy_norm(96), z_t(66)]) = 162` (symmetric; no asymmetry since this is sim-only)
- Actor MLP: `Linear(162->512->256->128->16)` + ELU. Last layer orthogonal init with `gain=0.01` -> initial action near 0 -> `q_nominal` light-close grasp.
- Critic MLP: `Linear(162->512->256->128->1)` + ELU.
- Actor output has **no tanh**; RSL-RL samples from `Normal(mean, std)`, and `GraspRelativeAction` clamps to `[-1, 1]`.

### 11.5 MoE interfaces (stubbed, not trained yet)

- `export_encoders()` returns deep copies of four state dicts: `actor_pointnet`, `actor_mu_e`, `critic_pointnet`, `critic_mu_e`.
- `load_encoders(state)` restores them (used by expert / gate stages).
- `freeze_encoders()` sets `requires_grad=False` on them.
- Gate input (MoE stage): `cat([f_t(32), size_encoding(5)], -1) = 37`.
