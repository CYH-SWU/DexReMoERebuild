# DexReMoERebuild

Reproduction of the base-policy stage of **DexReMoE: In-hand Reorientation of General Object via Mixtures of Experts**.

The paper proposes a Mixture-of-Experts policy for dexterous in-hand object reorientation. This repository rebuilds the environment, the network, and the training / evaluation loop for the **base policy** stage, using the Allegro hand and the HORA grasp cache. The MoE stage (expert policies + soft gate) is not implemented; interfaces are stubbed so it can be added later.

See `docs/` for the full design.

---

## Paper

- Title: *DexReMoE: In-hand Reorientation of General Object via Mixtures of Experts*
- Authors: Jun Wan, Xing Liu, Yunlong Dong
- Project page: https://wj-0212.github.io/

Key idea: instead of training a single monolithic policy for all objects, the paper trains multiple size-specialised experts and adds a small gating network that softly routes each object to the experts based on its geometry.

---

## Data

### Source

The grasp cache comes from the dataset that ships with the HORA in-hand rotation codebase:

- Repository: `Wonik-Robotics/allegro_inhand_rotation`
- Object set: 11 scale factors `s066, s07, s072, s074, s076, s078, s08, s082, s084, s086, s09`, each with 50,000 grasps
- Objects: three simple primitives (cylinder, cuboid, sphere) scaled by the scale factor. Characteristic object size is `0.08 m * scale` (s08 sphere is 6.4 cm in diameter, s066 is 5.3 cm, s09 is 7.2 cm).

Raw file layout: `assets/grasp_cache/cache/allegro_right_grasp_50k_<scale>.npy`, shape `(50000, 23)`, float32. Columns:

| columns | content | frame |
|---|---|---|
| `[0:16]` | Allegro joint angles (rad) | HORA / Isaac Gym DOF order |
| `[16:19]` | object position (x, y, z) | HORA world frame |
| `[19:23]` | object quaternion (x, y, z, w) | HORA world frame |

Data is **not committed** to the repo (see `.gitignore`); download the raw cache separately and place it under `assets/grasp_cache/cache/`.

### Processing pipeline

The cache goes through three stages. Each stage writes a separate directory so every intermediate result stays reproducible.

**Stage 1 - preprocess** (`scripts/preprocess_grasp.py`)

    raw .npy  ->  assets/grasp_cache_isaac/{train,test}/grasp_<scale>.npz

- Remaps joints from HORA order to IsaacLab Allegro USD order with the fixed permutation `[0, 8, 12, 4, 1, 9, 13, 5, 2, 10, 14, 6, 3, 11, 15, 7]`.
- Rotates the object pose from HORA frame to IsaacLab frame.
- Replaces the raw object position with the fingertip centroid obtained from forward kinematics (so the object always sits at the geometric centre of the four fingertips).
- Samples per-row physical parameters: `mass ~ U(0.02, 0.10) kg`, `friction ~ U(0.5, 1.5)`, `com_offset ~ U(-0.01, 0.01)^3 m`, `shape_id` (random label, not used downstream).

**Stage 2 - stability validation** (`scripts/validate_grasp_stability.py`, batch driver `scripts/validate_all_single_process.py`)

    assets/grasp_cache_isaac/  ->  assets/grasp_cache_isaac_validated/

For each (scale, shape) group, load the cache rows into a parallel Allegro scene, run 60 physics steps with gravity on, and keep only the rows where the object does not drift or fall below tolerance:

    pos_drift  < 0.25 * size
    ori_drift  < 30 deg
    z_drop     < 0.50 * size
    z_final    < 0.40 * size

Typical pass rate is 9 - 16 %, so the validated pool has ~58,753 rows across 27 train files (9 scales x 3 shapes) and ~13,940 rows across 6 test files (2 scales x 3 shapes).

**Stage 3 - training cache** (`source/dexremoe_tasks/cache/grasp_cache.py`)

The training env reads the validated `.npz` files directly. `GraspCache` supports:

- stratified 1/3 sampling across small / medium / large size buckets (`data_bucket=None`, used by base and gate),
- single-bucket sampling (`data_bucket="small" | "medium" | "large"`, used by experts),
- deterministic without-replacement sampling for eval (`env.eval_mode=True`, fixed mapping `env i -> eval_pool[(eval_offset + i) % n_eval]`).

The file suffix `cuboid` / `cylinder` / `sphere` is the authoritative geometry. The `shape_id` column is a random label produced by the preprocessing step and must **not** be used to pick geometry or enter observations.

---

## Repository layout

    DexReMoERebuild/
    +-- README.md
    +-- docs/
    |   +-- env_design.md        env, observation, reward, network design
    |   +-- training.md          PPO config, tuning history, TensorBoard guide
    +-- scripts/
    |   +-- preprocess_grasp.py             stage 1
    |   +-- validate_grasp_stability.py     stage 2, single group
    |   +-- validate_all_single_process.py  stage 2, batch (one Kit process)
    |   +-- visualize_grasp.py              render a raw grasp from the dataset
    |   +-- train_base.py                   base-policy PPO training
    |   +-- eval_base.py                    load a checkpoint, run eval episodes
    |   +-- play_base.py                    load a checkpoint, render
    |   +-- play_env.py                     render the env with zero action
    +-- source/
    |   +-- dexremoe_tasks/      env package (no networks)
    |   +-- dexremoe_agents/     network package (no env)
    +-- assets/
        +-- grasp_cache/                     raw .npy cache (not committed)
        +-- grasp_cache_isaac/               stage 1 output
        +-- grasp_cache_isaac_validated/     stage 2 output

---

## Quick start

All scripts assume the `isaaclab` conda environment (Isaac Sim 5.1 + IsaacLab 2.2.0). The two packages must be installed editable once:

    pip install -e DexReMoERebuild/source/dexremoe_tasks --no-deps
    pip install -e DexReMoERebuild/source/dexremoe_agents --no-deps

### 1. Preprocess the raw cache

    python DexReMoERebuild/scripts/preprocess_grasp.py \
        --cache_dir DexReMoERebuild/assets/grasp_cache/cache \
        --output_dir DexReMoERebuild/assets/grasp_cache_isaac \
        --split all --num_rows 50000 --settle_steps 0

### 2. Validate stability

    python DexReMoERebuild/scripts/validate_all_single_process.py --num_envs 256

### 3. Train base policy

    python DexReMoERebuild/scripts/train_base.py \
        --num_envs 64 --max_iter 1000 --save_interval 100 \
        --run_name my_run

Checkpoints and TensorBoard events land in `logs/rsl_rl/base_policy/my_run/`. TensorBoard:

    tensorboard --logdir DexReMoERebuild/logs/rsl_rl/base_policy

### 4. Evaluate and render

    python DexReMoERebuild/scripts/eval_base.py \
        --checkpoint logs/rsl_rl/base_policy/my_run/model_final.pt

    python DexReMoERebuild/scripts/play_base.py \
        --checkpoint logs/rsl_rl/base_policy/my_run/model_final.pt

---

## Status and known gaps

- The env, the networks, and the training / eval loop are functional.
- Base policy training runs end to end. On an 8 GiB laptop GPU the budget is ~1e6 to 5e6 timesteps, far below the paper scale (5e8+). Expect a stable positive trend, not paper-level numbers.
- MoE (expert policies + gate) is not implemented. The interfaces (`data_bucket`, `eval_mode`, `export_encoders`, `freeze_encoders`, `size_encoding`) are in place so it can be added without touching the env.
- The cache was validated against a free-floating hand; training uses a fixed hand. See `docs/env_design.md` section 10 for the accepted ~75% replay gap.
