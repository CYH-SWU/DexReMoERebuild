"""Visualize an Allegro grasp from the Wonik/HORA dataset in IsaacLab.

Dataset layout (confirmed from HORA source allegro_hand_hora.py):
    cols [0:16]  -> Allegro joint positions (rad), HORA/URDF order
    cols [16:19] -> object position (x, y, z), HORA world frame
    cols [19:23] -> object quaternion (qx, qy, qz, qw), HORA world frame

HORA URDF order (per-finger):
    [idx0,idx1,idx2,idx3, mid0,mid1,mid2,mid3, rng0,rng1,rng2,rng3, thb0,thb1,thb2,thb3]

IsaacLab Allegro USD order (per-level, confirmed from logs/joint_order.json):
    [idx0,mid0,rng0,thb0, idx1,mid1,rng1,thb1, idx2,mid2,rng2,thb2, idx3,mid3,rng3,thb3]

HORA / Isaac Gym DOF order (from HORA source comment):
    [0:4]   index  (joint_0.0  .. joint_3.0)
    [4:8]   thumb  (joint_12.0 .. joint_15.0)
    [8:12]  middle (joint_4.0  .. joint_7.0)
    [12:16] ring   (joint_8.0  .. joint_11.0)

So the permutation HORA -> IsaacLab is:
    isaac[i] = hora[PERM[i]]
    PERM = [0, 8, 12, 4, 1, 9, 13, 5, 2, 10, 14, 6, 3, 11, 15, 7]
"""

import argparse
import json
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
from scipy.spatial.transform import Rotation as Rot

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

# Workaround: pip Isaac Sim 5.1 ships without apps/rendering_modes/*.kit
SimulationContext._apply_render_settings_from_cfg = lambda self: None


LOG_PATH = "/home/chen/my_project/DexReMoERebuild/logs/visualize_grasp.log"
JSON_PATH = "/home/chen/my_project/DexReMoERebuild/logs/visualize_grasp_result.json"

# HORA / Isaac Gym DOF order (from HORA source comment):
#   [0:4]   index  (joint_0.0  .. joint_3.0)
#   [4:8]   thumb  (joint_12.0 .. joint_15.0)
#   [8:12]  middle (joint_4.0  .. joint_7.0)
#   [12:16] ring   (joint_8.0  .. joint_11.0)
#
# IsaacLab order: idx0, mid0, rng0, thb0, idx1, mid1, rng1, thb1, ...
#
# isaac[i] = hora[PERM[i]]
HORA_TO_ISAAC_PERM = np.array([0, 8, 12, 4, 1, 9, 13, 5, 2, 10, 14, 6, 3, 11, 15, 7], dtype=np.int64)


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


@configclass
class GraspVizSceneCfg(InteractiveSceneCfg):
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


def wxyz_to_rot(q_wxyz):
    return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


# Hand base poses (both at the same world origin, different orientation)
HAND_POS = np.array([0.0, 0.0, 0.5], dtype=np.float64)
Q_HORA_WXYZ = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)
Q_ISAAC_WXYZ = np.array(
    [0.25755104422569275, 0.28304505348205566, 0.6833301782608032, -0.621782124042511],
    dtype=np.float64,
)


def load_grasp(npy_path: str, row_idx: int):
    """Load one grasp: joint angles (permuted) and the object orientation.

    We do NOT transform the object position through the hand base frame here.
    HORA's base +x points toward the fingers, while IsaacLab's USD base frame
    has a different semantic (fingers point along -y), so a base-frame
    transform silently maps the object to the wrong side of the hand. Instead,
    the caller places the object at the actual fingertip centroid after the
    fingers have been driven to the grasp pose.

    Object orientation: the dataset quaternion is expressed in the HORA world
    frame. We apply the same R_delta = R_isaac * R_hora^{-1} to re-express it
    in the IsaacLab world frame, because orientation is a pure rotation and
    does not depend on where the base origin sits.
    """
    data = np.load(npy_path, allow_pickle=False)
    log(f"loaded {npy_path}, shape={data.shape}, dtype={data.dtype}")
    row = data[row_idx].copy()

    joint_hora = row[:16].astype(np.float32)
    joint_isaac = joint_hora[HORA_TO_ISAAC_PERM]

    obj_quat_hora_xyzw = row[19:23].astype(np.float64)

    R_hora = wxyz_to_rot(Q_HORA_WXYZ)
    R_isaac = wxyz_to_rot(Q_ISAAC_WXYZ)
    R_delta = R_isaac * R_hora.inv()

    R_obj_hora = Rot.from_quat(obj_quat_hora_xyzw)
    R_obj_isaac = R_delta * R_obj_hora
    q_isaac_xyzw = R_obj_isaac.as_quat()
    obj_quat_isaac_wxyz = np.array(
        [q_isaac_xyzw[3], q_isaac_xyzw[0], q_isaac_xyzw[1], q_isaac_xyzw[2]],
        dtype=np.float32,
    )

    log(f"joint_hora : {joint_hora}")
    log(f"joint_isaac: {joint_isaac}")
    log(f"obj_quat_hora (wxyz): {[obj_quat_hora_xyzw[3], *obj_quat_hora_xyzw[:3]]}")
    log(f"obj_quat_isaac(wxyz): {obj_quat_isaac_wxyz}")

    return joint_isaac, obj_quat_isaac_wxyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy",
        type=str,
        default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache/cache/allegro_right_grasp_50k_s08.npy",
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log("=== visualize_grasp start ===")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(GraspVizSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim.play()

    robot = scene["robot"]
    obj = scene["object"]

    log(f"joint_names: {robot.joint_names}")
    log(f"robot root pos: {robot.data.root_pos_w[0].tolist()}")
    log(f"robot root quat: {robot.data.root_quat_w[0].tolist()}")
    log(f"body_names: {robot.body_names}")

    joint_pos_np, obj_quat_wxyz_np = load_grasp(args.npy, args.row)

    # 1. write joint state AND target (prevent PD pull-back)
    joint_pos_t = torch.tensor(joint_pos_np, device="cuda:0").unsqueeze(0)
    joint_vel_t = torch.zeros_like(joint_pos_t)
    robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t)
    robot.set_joint_position_target(joint_pos_t)
    scene.write_data_to_sim()

    # 2. let the fingers settle at the grasp pose before placing the object
    fingertip_body_names = [
        "index_link_3",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    ]
    body_ids = [robot.body_names.index(n) for n in fingertip_body_names if n in robot.body_names]
    log(f"fingertip body ids: {body_ids} for {fingertip_body_names}")

    for _ in range(20):
        sim.step()
        scene.update(sim.get_physics_dt())

    # 3. compute fingertip centroid and place the object there
    body_pos = robot.data.body_pos_w[0]
    tips = [body_pos[bid].cpu().numpy() for bid in body_ids]
    tip_centroid = np.mean(tips, axis=0)
    log(f"fingertip positions: {[np.round(t, 4).tolist() for t in tips]}")
    log(f"fingertip centroid: {np.round(tip_centroid, 4).tolist()}")

    root_state = torch.zeros((1, 13), device="cuda:0")
    root_state[0, 0:3] = torch.tensor(tip_centroid, device="cuda:0", dtype=torch.float32)
    root_state[0, 3:7] = torch.tensor(obj_quat_wxyz_np, device="cuda:0", dtype=torch.float32)
    obj.write_root_state_to_sim(root_state)
    scene.write_data_to_sim()

    log(f"placed object at fingertip centroid: {np.round(tip_centroid, 4).tolist()}")

    for i in range(args.steps):
        sim.step()
        scene.update(sim.get_physics_dt())
        if i % 30 == 0 or i == args.steps - 1:
            obj_pos_now = obj.data.root_pos_w[0].tolist()
            hand_pos = robot.data.root_pos_w[0].tolist()
            tip_positions = {}
            if body_ids:
                body_pos = robot.data.body_pos_w[0]
                for name, bid in zip(fingertip_body_names, body_ids):
                    tip_positions[name] = body_pos[bid].tolist()
            log(f"step {i:03d}: hand_base={hand_pos}, obj={obj_pos_now}")
            for name, p in tip_positions.items():
                log(f"    {name}: {p}")

    # 4. dump final snapshot for offline analysis
    snapshot = {
        "hand_base": robot.data.root_pos_w[0].tolist(),
        "hand_quat": robot.data.root_quat_w[0].tolist(),
        "object": obj.data.root_pos_w[0].tolist(),
        "joint_pos_isaac": joint_pos_np.tolist(),
        "joint_pos_actual": robot.data.joint_pos[0].tolist(),
        "fingertips": {n: robot.data.body_pos_w[0][b].tolist() for n, b in zip(fingertip_body_names, body_ids)},
    }
    with open(JSON_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)
    log(f"wrote snapshot to {JSON_PATH}")

    log("=== holding pose (close window to exit) ===")
    while True:
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
