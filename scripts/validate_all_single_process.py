"""Stability validation: single Kit process, loop over all groups.

Why: spawning one Kit process per group accumulated global state and crashed
at group 7 (USD stage GetPrimAtPath(None)). Here we start Kit once, create a
single SimulationContext, and for each group delete /World/envs, rebuild the
InteractiveScene, validate, and move on.

Fixed PD: stiffness=3.0, damping=1.0.
Thresholds scale with object size (HORA base 0.08 m * scale):
  pos_tol = 0.25 * size, ori_tol = 30 deg,
  z_drop_tol = 0.50 * size, z_final_tol = 0.40 * size

Outputs (same as per-group script):
  <output_dir>/<split>/grasp_<scale>_<shape>.npz     filtered rows
  <log_dir>/per_sample_<split>_<scale>_<shape>.npz  per-row metrics
  <log_dir>/summaries.jsonl                          one SUMMARY_JSON per group
"""

import argparse
import gc
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
import omni.usd

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
STEPS = 60

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

def build_scene_cfg(shape_id: int, size: float, num_envs: int):
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
            spawn=make_spawn_cfg(shape_id, size),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    return SceneCfg(num_envs=num_envs, env_spacing=2.0)

def clear_world_envs():
    """Remove everything under /World except /World itself.

    InteractiveScene spawns ground, light, and per-env prims. On the next
    group we must remove all of them, otherwise spawn_ground_plane raises
    "A prim already exists at path: '/World/ground'".
    """
    stage = omni.usd.get_context().get_stage()
    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        for child in list(world.GetChildren()):
            stage.RemovePrim(child.GetPath())

def validate_group(
    tag, npz_path, shape_id, split, scale_name,
    output_dir, log_dir, sim, num_envs, max_rows,
):
    d = np.load(npz_path)
    scale = float(d["scale"][0])
    size = HORA_BASE_SIZE * scale
    pos_tol = 0.25 * size
    ori_tol = 30.0
    z_drop_tol = 0.50 * size
    z_final_tol = 0.40 * size

    shape_mask = d["shape_id"] == shape_id
    idx_all = np.where(shape_mask)[0]
    if max_rows > 0:
        idx_all = idx_all[:max_rows]
    n_total = len(idx_all)

    print(f"[{tag}] total={n_total} size={size*100:.2f}cm "
          f"pos_tol={pos_tol*100:.2f}cm z_drop={z_drop_tol*100:.2f}cm "
          f"z_final={z_final_tol*100:.2f}cm", flush=True)
    if n_total == 0:
        print(f"[{tag}] no rows, skipping", flush=True)
        return None

    # rebuild scene
    clear_world_envs()
    scene = InteractiveScene(build_scene_cfg(shape_id, size, num_envs))
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

    n = n_total
    per_row = {
        "row": np.zeros(n, dtype=np.int64),
        "passed": np.zeros(n, dtype=bool),
        "pos_drift": np.zeros(n, dtype=np.float32),
        "ori_drift_deg": np.zeros(n, dtype=np.float32),
        "z_drop": np.zeros(n, dtype=np.float32),
        "z_final_drop": np.zeros(n, dtype=np.float32),
        "reason_mask": np.zeros(n, dtype=np.uint8),
    }

    t0 = time.time()
    B = num_envs
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
        for _ in range(STEPS):
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
        if (start // B) % 10 == 0 or end == n:
            print(f"[{tag}] {end}/{n} | pass={n_pass} ({n_pass/end*100:.1f}%) | "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    per_sample_path = os.path.join(
        log_dir, f"per_sample_{split}_{scale_name}_{SHAPE_NAMES[shape_id]}.npz"
    )
    np.savez_compressed(per_sample_path, **per_row)

    passed_mask = per_row["passed"]
    passed_src_rows = per_row["row"][passed_mask]
    out_path = os.path.join(output_dir, split, f"grasp_{scale_name}_{SHAPE_NAMES[shape_id]}.npz")
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

    # drop scene references so next iteration can rebuild cleanly
    del robot, obj, scene
    gc.collect()
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated scale names to run, e.g. s074. Empty = all.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)

    train_scales = ["s07", "s072", "s074", "s076", "s078", "s08", "s082", "s084", "s086"]
    test_scales = ["s066", "s09"]
    groups = [(s, "train") for s in train_scales] + [(s, "test") for s in test_scales]
    if args.only:
        keep = set(x.strip() for x in args.only.split(",") if x.strip())
        groups = [g for g in groups if g[0] in keep]

    print(f"[main] groups={len(groups)}: {[g[0] for g in groups]}", flush=True)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    # a tiny bootstrap scene is not needed; scene gets created per group

    summary_path = os.path.join(args.log_dir, "summaries.jsonl")
    start_all = time.time()
    with open(summary_path, "a") as fsum:
        for scale, split in groups:
            npz = os.path.join(args.source_dir, split, f"grasp_{scale}.npz")
            if not os.path.isfile(npz):
                print(f"[main] skip missing {npz}", flush=True)
                continue
            for shape_id in (0, 1, 2):
                tag = f"{split}/{scale}/{SHAPE_NAMES[shape_id]}"
                print(f"===== group: {tag} =====", flush=True)
                try:
                    s = validate_group(
                        tag, npz, shape_id, split, scale,
                        args.output_dir, args.log_dir, sim,
                        args.num_envs, args.max_rows,
                    )
                    if s is not None:
                        fsum.write(f"SUMMARY_JSON {json.dumps(s)}\n")
                        fsum.flush()
                except Exception as e:
                    import traceback
                    print(f"[{tag}] FAILED: {e}", flush=True)
                    traceback.print_exc()
                    # try to recover by clearing the world so next group starts clean
                    try:
                        clear_world_envs()
                    except Exception:
                        pass

    print(f"[main] ALL DONE in {time.time()-start_all:.1f}s", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
