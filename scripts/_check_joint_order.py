"""Print IsaacLab Allegro joint order and per-joint axis info."""
import json

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/cloud",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

SimulationContext._apply_render_settings_from_cfg = lambda self: None


@configclass
class Cfg(InteractiveSceneCfg):
    robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    out = {
        "num_joints": robot.num_joints,
        "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "root_pos": robot.data.root_pos_w[0].tolist(),
        "root_quat_wxyz": robot.data.root_quat_w[0].tolist(),
    }

    # joint limits if available
    try:
        out["joint_limits_lower"] = robot.data.joint_limits[0, :, 0].tolist()
        out["joint_limits_upper"] = robot.data.joint_limits[0, :, 1].tolist()
    except Exception as e:
        out["joint_limits_error"] = str(e)

    with open("/home/chen/my_project/DexReMoERebuild/logs/joint_order.json", "w") as f:
        json.dump(out, f, indent=2)

    app.close()


if __name__ == "__main__":
    main()
