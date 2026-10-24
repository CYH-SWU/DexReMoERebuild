"""Offline preprocessing: convert HORA grasp cache to IsaacLab-ready cache.

For each selected row of a HORA .npy file, produce one entry with:
  - joint_isaac     : (16,)  joint angles in IsaacLab order
  - obj_pos_isaac   : (3,)   object position = fingertip centroid (IsaacLab world)
  - obj_quat_wxyz   : (4,)   object orientation after R_delta rotation
  - shape_id        : scalar, 0=cylinder, 1=cuboid, 2=sphere (random, p=0.35/0.2/0.45)
  - scale           : scalar, parsed from the filename (s08 -> 0.80)
  - mass            : scalar, U(0.02, 0.10) kg
  - friction        : scalar, U(0.5, 1.5)
  - com_offset      : (3,)   center-of-mass offset in the object local frame, U(-0.01, 0.01)^3 m

Usage:
    python preprocess_grasp.py \
        --npy assets/grasp_cache/cache/allegro_right_grasp_50k_s08.npy \
        --row 0 --num_rows 1 \
        --output assets/grasp_cache_isaac/train/grasp_s08.npz
"""

import argparse
import os
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/cloud",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
)

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

# pip Isaac Sim 5.1 ships without apps/rendering_modes/*.kit
SimulationContext._apply_render_settings_from_cfg = lambda self: None


# HORA / Isaac Gym DOF order -> IsaacLab order
# HORA:  index(0-3), thumb(4-7), middle(8-11), ring(12-15)
# Isaac: idx0, mid0, rng0, thb0, idx1, mid1, ...
HORA_TO_ISAAC_PERM = np.array([0, 8, 12, 4, 1, 9, 13, 5, 2, 10, 14, 6, 3, 11, 15, 7], dtype=np.int64)

Q_HORA_WXYZ = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)
Q_ISAAC_WXYZ = np.array(
    [0.25755104422569275, 0.28304505348205566, 0.6833301782608032, -0.621782124042511],
    dtype=np.float64,
)

FINGERTIP_BODY_NAMES = [
    "index_link_3",
    "middle_biotac_tip",
    "ring_biotac_tip",
    "thumb_biotac_tip",
]


def wxyz_to_rot(q_wxyz):
    return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def parse_scale(npy_path: str) -> float:
    """Parse scale from filename.

    s07 -> 0.70, s08 -> 0.80, s09 -> 0.90
    s066 -> 0.66, s072 -> 0.72, ..., s086 -> 0.86
    """
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    s = stem.split("_s")[-1]
    # `s0X` means 0.X0 (e.g. s08 -> 0.80), while `sXYZ` is 0.XYZ.
    if len(s) == 2 and s[0] == "0":
        s = s[1] + "0"
    return int(s) / 100.0


@configclass
class PreprocessSceneCfg(InteractiveSceneCfg):
    """Minimal scene: one Allegro hand and one placeholder object.

    The object is only needed so the scene is valid; it is not placed or
    touched during preprocessing. Its size is set from the scale so that if
    the scene is later reused for validation, the geometry is consistent.
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )

    robot = ALLEGRO_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.5,
                stiffness=100.0,
                damping=2.0,
                friction=0.01,
            ),
        },
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )


def setup_scene():
    """Create the IsaacLab scene once and return (sim, scene, robot, body_ids)."""
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(PreprocessSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim.play()

    robot = scene["robot"]
    body_ids = [robot.body_names.index(nm) for nm in FINGERTIP_BODY_NAMES if nm in robot.body_names]
    if len(body_ids) != len(FINGERTIP_BODY_NAMES):
        raise RuntimeError(
            f"Missing fingertip bodies: expected {FINGERTIP_BODY_NAMES}, "
            f"got {robot.body_names}"
        )
    return sim, scene, robot, body_ids


def preprocess(npy_path: str, row_start: int, num_rows: int, settle_steps: int,
               sim, scene, robot, body_ids):
    data = np.load(npy_path, allow_pickle=False)
    total = data.shape[0]
    row_end = min(row_start + num_rows, total)
    rows = data[row_start:row_end].copy()
    n = rows.shape[0]
    print(f"[preprocess] file={npy_path} total={total} rows=[{row_start}:{row_end}] n={n}")

    scale = parse_scale(npy_path)
    print(f"[preprocess] scale={scale:.2f}")

    # --- joint angles ---
    joint_hora = rows[:, :16].astype(np.float32)
    joint_isaac = joint_hora[:, HORA_TO_ISAAC_PERM]

    # --- object orientation ---
    obj_quat_hora_xyzw = rows[:, 19:23].astype(np.float64)
    R_hora = wxyz_to_rot(Q_HORA_WXYZ)
    R_isaac = wxyz_to_rot(Q_ISAAC_WXYZ)
    R_delta = R_isaac * R_hora.inv()
    R_obj_hora = Rot.from_quat(obj_quat_hora_xyzw)
    R_obj_isaac = R_delta * R_obj_hora
    q_isaac_xyzw = R_obj_isaac.as_quat()
    obj_quat_wxyz = np.concatenate([q_isaac_xyzw[:, 3:4], q_isaac_xyzw[:, :3]], axis=1).astype(np.float32)

    obj_pos_isaac = np.zeros((n, 3), dtype=np.float32)
    timing_path = f"/tmp/preprocess_timing_{os.getpid()}.log"
    loop_start = time.time()
    for i in range(n):
        q = torch.tensor(joint_isaac[i : i + 1], device="cuda:0", dtype=torch.float32)
        qv = torch.zeros_like(q)
        robot.write_joint_state_to_sim(q, qv)
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()

        for _ in range(settle_steps):
            sim.step()
            scene.update(sim.get_physics_dt())

        tips = torch.stack([robot.data.body_pos_w[0][bid] for bid in body_ids])  # (4, 3)
        centroid = tips.mean(dim=0).cpu().numpy().astype(np.float32)
        obj_pos_isaac[i] = centroid

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - loop_start
            per_item = elapsed / (i + 1)
            msg = (
                f"[timing] processed {i+1}/{n} | elapsed={elapsed:.2f}s | "
                f"per_item={per_item*1000:.2f}ms | eta_total={per_item*n:.1f}s"
            )
            with open(timing_path, "a") as f:
                f.write(msg + "\n")
            print(msg, flush=True)

    # --- sampling: shape, mass, friction, com ---
    rng = np.random.default_rng(42 + row_start)
    shape_id = rng.choice(3, size=n, p=[0.35, 0.2, 0.45]).astype(np.int8)
    mass = rng.uniform(0.02, 0.10, size=n).astype(np.float32)
    friction = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    com_offset = rng.uniform(-0.01, 0.01, size=(n, 3)).astype(np.float32)
    scale_arr = np.full(n, scale, dtype=np.float32)

    return {
        "joint_isaac": joint_isaac.astype(np.float32),
        "obj_pos_isaac": obj_pos_isaac,
        "obj_quat_wxyz": obj_quat_wxyz,
        "shape_id": shape_id,
        "scale": scale_arr,
        "mass": mass,
        "friction": friction,
        "com_offset": com_offset,
    }


TRAIN_SCALES = ["s07", "s072", "s074", "s076", "s078", "s08", "s082", "s084", "s086"]
TEST_SCALES = ["s066", "s09"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache/cache",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache_isaac",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--num_rows", type=int, default=50000,
                        help="Rows per file. Use -1 for all rows.")
    parser.add_argument("--settle_steps", type=int, default=0)
    args = parser.parse_args()

    if args.split == "train":
        scales = [(s, "train") for s in TRAIN_SCALES]
    elif args.split == "test":
        scales = [(s, "test") for s in TEST_SCALES]
    else:
        scales = [(s, "train") for s in TRAIN_SCALES] + [(s, "test") for s in TEST_SCALES]

    for _, sub in scales:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    print(f"[main] processing {len(scales)} files, split={args.split}")
    for s, _ in scales:
        print(f"[main]   {s}")

    sim, scene, robot, body_ids = setup_scene()
    print(f"[main] scene ready, body_ids={body_ids}")

    global_start = time.time()
    for idx, (scale_name, sub) in enumerate(scales):
        npy_path = os.path.join(
            args.cache_dir, f"allegro_right_grasp_50k_{scale_name}.npy"
        )
        out_path = os.path.join(args.output_dir, sub, f"grasp_{scale_name}.npz")

        num_rows = args.num_rows
        if num_rows < 0:
            num_rows = 10**9

        file_start = time.time()
        result = preprocess(
            npy_path, 0, num_rows, args.settle_steps,
            sim, scene, robot, body_ids,
        )
        np.savez(out_path, **result)
        file_elapsed = time.time() - file_start
        total_elapsed = time.time() - global_start
        print(
            f"[main] [{idx+1}/{len(scales)}] saved {result['joint_isaac'].shape[0]} rows "
            f"to {out_path} in {file_elapsed:.1f}s (total {total_elapsed:.1f}s)",
            flush=True,
        )

    print(f"[main] ALL DONE in {time.time() - global_start:.1f}s")
    app.close()


if __name__ == "__main__":
    main()
