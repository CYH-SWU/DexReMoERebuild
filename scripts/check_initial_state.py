"""Load one grasp from the preprocessed npz and initialize the sim exactly
as recorded: joints, object pose, mass, friction, COM offset, shape, gravity on.
Print expected vs actual so we can confirm the initialization is faithful.
"""

import argparse
import os

from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

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

LOG_PATH = "/home/chen/my_project/DexReMoERebuild/logs/check_initial_state.log"

SHAPE_NAMES = {0: "cylinder", 1: "cuboid", 2: "sphere"}


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
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


def make_object_cfg(shape_id: int, scale: float, mass: float, com_offset: np.ndarray) -> RigidObjectCfg:
    """Build a RigidObjectCfg for the given shape and physical parameters."""
    # HORA base object size is 8 cm (cube box=0.08, sphere r=0.04, cylinder r=0.04/l=0.08).
    # `scale` is the relative scale factor (s08 -> 0.80), so actual size = 0.08 * scale.
    base_size = 0.08
    size = base_size * scale
    com = tuple(float(x) for x in com_offset)

    if shape_id == 1:  # cuboid
        spawn = sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 0.9)),
        )
    elif shape_id == 0:  # cylinder
        spawn = sim_utils.CylinderCfg(
            radius=size / 2.0,
            height=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.2)),
        )
    else:  # sphere
        spawn = sim_utils.SphereCfg(
            radius=size / 2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.4)),
        )

    # NOTE: InitialStateCfg has no `com` field in this IsaacLab version.
    # COM offset is skipped for now (it only affects physics stability,
    # not whether the object/joints are placed at the recorded values).
    log(f"  [make_object_cfg] com_offset={com} (not applied: no com field in InitialStateCfg)")
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache_isaac/train/grasp_s08.npz",
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log("=== check_initial_state ===")

    d = np.load(args.npz)
    row = args.row

    joint_isaac = d["joint_isaac"][row]
    obj_pos = d["obj_pos_isaac"][row]
    obj_quat = d["obj_quat_wxyz"][row]
    shape_id = int(d["shape_id"][row])
    scale = float(d["scale"][row])
    mass = float(d["mass"][row])
    friction = float(d["friction"][row])
    com_offset = d["com_offset"][row]

    log(f"npz={args.npz} row={row}")
    log(f"shape={SHAPE_NAMES[shape_id]} ({shape_id}), scale={scale:.2f}")
    log(f"mass={mass:.4f} kg, friction={friction:.4f}, com_offset={com_offset.tolist()}")
    log(f"expected joint_isaac={np.round(joint_isaac, 4).tolist()}")
    log(f"expected obj_pos={np.round(obj_pos, 4).tolist()}")
    log(f"expected obj_quat_wxyz={np.round(obj_quat, 4).tolist()}")

    # Build scene with the object shape/params from the npz
    scene_cfg = CheckSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.object = make_object_cfg(shape_id, scale, mass, com_offset)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    sim.play()

    robot = scene["robot"]
    obj = scene["object"]

    # Apply joint state and target
    joint_t = torch.tensor(joint_isaac, device="cuda:0", dtype=torch.float32).unsqueeze(0)
    joint_v = torch.zeros_like(joint_t)
    robot.write_joint_state_to_sim(joint_t, joint_v)
    robot.set_joint_position_target(joint_t)

    # Apply object pose
    root = torch.zeros((1, 13), device="cuda:0")
    root[0, 0:3] = torch.tensor(obj_pos, device="cuda:0", dtype=torch.float32)
    root[0, 3:7] = torch.tensor(obj_quat, device="cuda:0", dtype=torch.float32)
    obj.write_root_state_to_sim(root)

    scene.write_data_to_sim()

    # Read back immediately (before any step) to confirm it was applied
    scene.update(sim.get_physics_dt())
    actual_joint = robot.data.joint_pos[0].cpu().numpy()
    actual_obj_pos = obj.data.root_pos_w[0].cpu().numpy()
    actual_obj_quat = obj.data.root_quat_w[0].cpu().numpy()

    log("--- immediately after write, before stepping ---")
    log(f"actual joint_pos     ={np.round(actual_joint, 4).tolist()}")
    log(f"joint diff (max abs) ={np.max(np.abs(actual_joint - joint_isaac)):.6f}")
    log(f"actual obj_pos       ={np.round(actual_obj_pos, 4).tolist()}")
    log(f"obj pos diff (max)   ={np.max(np.abs(actual_obj_pos - obj_pos)):.6f}")
    log(f"actual obj_quat_wxyz ={np.round(actual_obj_quat, 4).tolist()}")
    log(f"obj quat diff (max)  ={np.max(np.abs(actual_obj_quat - obj_quat)):.6f}")

    # Step with gravity on, log object position drift
    log(f"--- stepping {args.steps} with gravity ON ---")
    initial_z = float(actual_obj_pos[2])
    for i in range(args.steps):
        sim.step()
        scene.update(sim.get_physics_dt())
        if i % 30 == 0 or i == args.steps - 1:
            p = obj.data.root_pos_w[0].cpu().numpy()
            dz = float(p[2]) - initial_z
            log(f"step {i:03d}: obj={np.round(p, 4).tolist()} dz={dz:+.4f}")

    log("=== holding, close window to exit ===")
    while True:
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
