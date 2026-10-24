"""Full stability validation for one (shape, scale) group.

Reads every row of the given shape from the npz cache, initializes K parallel
environments with recorded joint angles / object pose / mass, enables gravity,
steps M steps, and checks whether the object stays near its initial pose.

Thresholds scale with object size (HORA base 0.08 m * scale):
  pos_tol    = 0.25 * size
  ori_tol    = 30 deg
  z_drop_tol = 0.50 * size
  z_final_tol= 0.40 * size

Fixed PD: stiffness=3.0, damping=1.0.

Outputs:
  --output_dir/<split>/grasp_<scale>.npz      filtered (passing rows only)
  --log_dir/per_sample_<split>_<scale>_<shape>.npz   per-row metrics
"""

import argparse
import json
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

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

SimulationContext._apply_render_settings_from_cfg = lambda self: None

SHAPE_NAMES = {0: "cylinder", 1: "cuboid", 2: "sphere"}
HORA_BASE_SIZE = 0.08
KP = 3.0
KD = 1.0


def make_spawn_cfg(shape_id: int, size: float):
    if shape_id == 1:
        return sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 0.9)),
        )
    if shape_id == 0:
        return sim_utils.CylinderCfg(
            radius=size / 2.0,
            height=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.2)),
        )
    return sim_utils.SphereCfg(
        radius=size / 2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.4)),
    )


def quat_angle_deg(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(q0_wxyz * q1_wxyz, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--shape_id", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--scale_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--max_rows", type=int, default=-1,
                        help="If >0, only validate the first N rows of this shape (for timing).")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, args.split), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    d = np.load(args.npz)
    scale = float(d["scale"][0])
    size = HORA_BASE_SIZE * scale
    pos_tol = 0.25 * size
    ori_tol = 30.0
    z_drop_tol = 0.50 * size
    z_final_tol = 0.40 * size

    shape_mask = d["shape_id"] == args.shape_id
    idx_all = np.where(shape_mask)[0]
    if args.max_rows > 0:
        idx_all = idx_all[: args.max_rows]
    n_total = len(idx_all)

    tag = f"{args.split}/{args.scale_name}/{SHAPE_NAMES[args.shape_id]}"
    print(f"[{tag}] total={n_total} size={size*100:.2f}cm "
          f"pos_tol={pos_tol*100:.2f}cm z_drop={z_drop_tol*100:.2f}cm "
          f"z_final={z_final_tol*100:.2f}cm", flush=True)

    if n_total == 0:
        print(f"[{tag}] no rows, skipping", flush=True)
        app.close()
        return

    @configclass
    class SceneCfg(InteractiveSceneCfg):
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
                    stiffness=KP,
                    damping=KD,
                    friction=0.01,
                ),
            },
        )
        object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=make_spawn_cfg(args.shape_id, size),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim.play()

    robot = scene["robot"]
    obj = scene["object"]

    mass_settable = True
    try:
        _ = obj.root_physx_view
    except Exception:
        mass_settable = False
        print(f"[{tag}] root_physx_view unavailable; using default mass", flush=True)

    n = n_total
    per_row = {
        "row": np.zeros(n, dtype=np.int64),
        "passed": np.zeros(n, dtype=bool),
        "pos_drift": np.zeros(n, dtype=np.float32),
        "ori_drift_deg": np.zeros(n, dtype=np.float32),
        "z_drop": np.zeros(n, dtype=np.float32),
        "z_final_drop": np.zeros(n, dtype=np.float32),
        "reason_mask": np.zeros(n, dtype=np.uint8),  # bit0 pos, bit1 ori, bit2 zmin, bit3 zfinal
    }

    t0 = time.time()
    B = args.num_envs
    for start in range(0, n, B):
        end = min(start + B, n)
        batch_idx = idx_all[start:end]
        k = len(batch_idx)

        joint = torch.tensor(d["joint_isaac"][batch_idx], device="cuda:0", dtype=torch.float32)
        pos = torch.tensor(d["obj_pos_isaac"][batch_idx], device="cuda:0", dtype=torch.float32)
        quat = torch.tensor(d["obj_quat_wxyz"][batch_idx], device="cuda:0", dtype=torch.float32)
        mass = torch.tensor(d["mass"][batch_idx], device="cuda:0", dtype=torch.float32)

        env_ids = torch.arange(k, device="cuda:0", dtype=torch.int64)
        env_origins = scene.env_origins[:k].to(device="cuda:0")

        robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=env_ids)
        robot.set_joint_position_target(joint, env_ids=env_ids)

        world_pos = pos + env_origins
        root = torch.zeros((k, 13), device="cuda:0")
        root[:, 0:3] = world_pos
        root[:, 3:7] = quat
        obj.write_root_state_to_sim(root, env_ids=env_ids)

        if mass_settable:
            try:
                masses_full = obj.root_physx_view.get_masses().clone().cpu()
                env_ids_cpu = env_ids.detach().cpu().int().contiguous()
                masses_full[env_ids_cpu] = mass.detach().cpu().float().unsqueeze(-1)
                obj.root_physx_view.set_masses(masses_full, env_ids_cpu)
            except Exception:
                mass_settable = False

        scene.write_data_to_sim()

        pos0 = world_pos.clone()
        z_min = pos0[:, 2].clone()

        for _ in range(args.steps):
            sim.step()
            scene.update(sim.get_physics_dt())
            z_min = torch.minimum(z_min, obj.data.root_pos_w[:k, 2])

        pos1 = obj.data.root_pos_w[:k].clone()
        quat1 = obj.data.root_quat_w[:k].clone()

        pos_drift = torch.norm(pos1 - pos0, dim=-1).cpu().numpy()
        ori_drift = quat_angle_deg(quat.cpu().numpy(), quat1.cpu().numpy())
        z_min_np = z_min.cpu().numpy()
        z_final_np = pos1[:, 2].cpu().numpy()
        z_init_np = pos0[:, 2].cpu().numpy()
        z_drop = z_init_np - z_min_np
        z_final_drop = z_init_np - z_final_np

        ok_pos = pos_drift < pos_tol
        ok_ori = ori_drift < ori_tol
        ok_zmin = z_drop < z_drop_tol
        ok_zfin = z_final_drop < z_final_tol
        passed = ok_pos & ok_ori & ok_zmin & ok_zfin

        reason = np.zeros(k, dtype=np.uint8)
        reason |= (~ok_pos).astype(np.uint8) << 0
        reason |= (~ok_ori).astype(np.uint8) << 1
        reason |= (~ok_zmin).astype(np.uint8) << 2
        reason |= (~ok_zfin).astype(np.uint8) << 3

        per_row["row"][start:end] = batch_idx
        per_row["passed"][start:end] = passed
        per_row["pos_drift"][start:end] = pos_drift
        per_row["ori_drift_deg"][start:end] = ori_drift
        per_row["z_drop"][start:end] = z_drop
        per_row["z_final_drop"][start:end] = z_final_drop
        per_row["reason_mask"][start:end] = reason

        n_pass = int(per_row["passed"][:end].sum())
        print(f"[{tag}] {end}/{n} | pass={n_pass} ({n_pass/end*100:.1f}%) | "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    # --- save per-sample metrics ---
    per_sample_path = os.path.join(
        args.log_dir, f"per_sample_{args.split}_{args.scale_name}_{SHAPE_NAMES[args.shape_id]}.npz"
    )
    np.savez_compressed(per_sample_path, **per_row)

    # --- save filtered cache ---
    passed_mask = per_row["passed"]
    passed_src_rows = per_row["row"][passed_mask]
    out_path = os.path.join(args.output_dir, args.split, f"grasp_{args.scale_name}_{SHAPE_NAMES[args.shape_id]}.npz")
    if len(passed_src_rows) > 0:
        np.savez(
            out_path,
            joint_isaac=d["joint_isaac"][passed_src_rows],
            obj_pos_isaac=d["obj_pos_isaac"][passed_src_rows],
            obj_quat_wxyz=d["obj_quat_wxyz"][passed_src_rows],
            shape_id=d["shape_id"][passed_src_rows],
            scale=d["scale"][passed_src_rows],
            mass=d["mass"][passed_src_rows],
            friction=d["friction"][passed_src_rows],
            com_offset=d["com_offset"][passed_src_rows],
            source_row=passed_src_rows.astype(np.int32),
        )

    n_pass = int(passed_mask.sum())
    summary = {
        "group": tag,
        "n_total": n,
        "n_pass": n_pass,
        "pass_rate": n_pass / n if n else 0.0,
        "object_size_cm": size * 100.0,
        "pos_tol_cm": pos_tol * 100.0,
        "z_drop_tol_cm": z_drop_tol * 100.0,
        "z_final_tol_cm": z_final_tol * 100.0,
        "elapsed_s": time.time() - t0,
        "per_sample_path": per_sample_path,
        "filtered_path": out_path if n_pass > 0 else None,
    }
    print(f"[{tag}] DONE n_pass={n_pass}/{n} ({n_pass/n*100:.1f}%)", flush=True)
    print(f"SUMMARY_JSON {json.dumps(summary)}", flush=True)

    app.close()


if __name__ == "__main__":
    main()
