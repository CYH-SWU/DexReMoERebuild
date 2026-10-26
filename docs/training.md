# Training Notes

## 1. Hardware ceiling

The paper trains with 32768 envs x 8 horizon = 262144 samples/iter. The local
machine is a single RTX 5060 Laptop with 8 GiB. The 33-prim in-hand scene alone
costs ~4.3 GiB, so the practical limit is 64 envs.

| config | samples / iter | GPU mem | runtime 1000 iter |
|---|---|---|---|
| 128 env, PointNet 1024 ch, 512 pts | 1024 | OOM at update | - |
| 128 env, PointNet 512 ch, 256 pts | 1024 | ~5.8 GiB | ~40 min |
| **64 env, PointNet 512 ch, 256 pts (current)** | **1536** | **~5.2 GiB** | **~40 min** |

64 envs is the largest that fits the training update without OOM.

## 2. PPO hyper-parameters

Current defaults in `scripts/train_base.py`:

| parameter | paper | ours | reason |
|---|---|---|---|
| learning_rate | 5e-3 | **3e-3** | 170x fewer samples/iter means 170x noisier gradient; 5e-3 makes KL routinely overshoot `desired_kl` and the adaptive scheduler halves it repeatedly. |
| num_steps_per_env (horizon) | 8 | **24** | 8 is much shorter than the GAE effective horizon (~20 at lam=0.95); value bootstrap is inaccurate and every episode gets split across 75 iterations. 24 matches the GAE horizon. |
| num_mini_batches | 16 (from 16384/262144) | **6** (auto: `max(1, min(16, num_envs*horizon//256))`) | Paper ratio assumes 262144 samples. At 1536 samples, 16 splits gives 96/batch, too noisy. Target ~256/batch. |
| clip_param | 0.2 | 0.2 | paper |
| gamma | 0.99 | 0.99 | paper |
| lam | 0.95 | 0.95 | paper |
| desired_kl | 0.02 | 0.02 | paper |
| entropy_coef | not given | 0.002 | RSL-RL default |
| num_learning_epochs | not given | 5 | RSL-RL default |

**Learning rate schedule** is RSL-RL's `"adaptive"`: it watches KL after each
update and halves lr if KL > 2 * desired_kl, doubles if KL < desired_kl / 2.
This is the callback; no cosine / warmup is added on top.

## 3. The "hold still" local optimum

With 64 envs and paper thresholds, after ~150 iterations we observed:

- `mean_episode_length` 50 -> 463 (policy learned to keep the grasp)
- `Episode_Termination/object_dropped` 0 (never drops)
- `success` still 0 (never aligned)
- `Metrics/object_pose/dtheta_deg` ~87 deg and rising
- `mean_reward` ~8.0 over 550 steps = +0.015 per step

The policy converged to "hold the object, do not rotate". Reason: at 87 deg the
orientation reward (+0.99/step) and the position penalty (-1.0/step) almost
cancel, so "do nothing" is a risk-free +0 reward. Attempting rotation risks
dropping (-800) or tripping action_rate.

## 4. Changes made to escape the local optimum

### Environment (DexReMoECommand)

| parameter | paper | ours | reason |
|---|---|---|---|
| `tau_theta` | 0.1 rad | **0.15 rad** | A fresh policy sits at ~87 deg; 0.1 rad is unreachable, success never fires. |
| `sustained_hold_steps` | 5 | **3** | Early policy is jittery; 5 consecutive perfect steps is essentially impossible. |

### Reward weights (`RewardsCfg`)

| term | paper | ours | reason |
|---|---|---|---|
| `orientation` | +1.0 | **+2.0** | Positive signal was being buried by accumulated negatives as episodes grew 10x. |
| `pos_penalty` | -10.0 | **-5.0** | At 500 steps, cumulative -10 * delta_p was dominating the reward. |
| `action_rate` | not in paper | **-0.005** | Halved from -0.01 to stop punishing early jittery exploration. |

The theoretical per-step reward range is now roughly [-0.65, +20] (excluding
+800 on success). Typical values are +0.8 to +3.0 for a partially aligned grasp.

### Diagnostic

`DexReMoECommand._update_metrics` now writes `self.metrics["dtheta_deg"]` so
TensorBoard shows the orientation error in degrees directly, avoiding the
back-solve from `Episode_Reward/orientation` (which divides by fixed 600 and
therefore inflates as episodes grow).

## 5. What to watch on TensorBoard

| metric | healthy | stuck | meaning |
|---|---|---|---|
| `Train/mean_episode_length` | rising toward 600 | flat below 100 | keeping the grasp |
| `Episode_Termination/object_dropped` | low | close to 1.0 | grasp stability |
| `Episode_Reward/success` | nonzero after ~300 iter | always 0 | escape local optimum |
| `Metrics/object_pose/max_consecutive_success` | > 0 | 0 | core metric |
| `Loss/kl_divergence` | 0.01 - 0.04 | > 0.05 | lr too high |
| `Train/mean_action_noise_std` | falling from 0.5 | flat at 0.5 | policy confidence |

Note `Episode_Reward/orientation` divides by fixed 600 (max episode length)
not by the actual length, so it inflates whenever episode length grows even if
the policy does not improve. Do not read it as "did the policy get better".

## 6. Known gaps

- Base policy is not expected to reach paper-level S (19.5) on this GPU. The
  budget is ~1e6 to 5e6 timesteps, the paper was 5e8+. Aim for a stable
  positive trend, not a number.
- Cache was validated with the hand free-floating; training uses a fixed hand.
  V2 stability check passes ~75% instead of ~100%. Accepted gap, see
  `docs/env_design.md` section 10.
- MoE (expert / gate) is not implemented. The env and the network already
  expose the interfaces (`data_bucket`, `eval_mode`, `export_encoders`,
  `freeze_encoders`, `size_encoding`). See `docs/env_design.md` sections 9 and
  11.5.
