"""DexReMoE base-policy env cfg."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..assets import ALLEGRO_DEXREMOE_CFG
from ..cache import GEOM_NAMES, SCALE_NAMES, SCALE_VALUES
import dexremoe_tasks.envs.mdp as dex_mdp


HORA_BASE_SIZE_M = 0.08


# ---------------------------------------------------------------- scene


def _make_obj_cfg(name: str, size: float, geom: str) -> RigidObjectCfg:
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    if geom == "cuboid":
        spawn = sim_utils.CuboidCfg(size=(size, size, size), **common)
    elif geom == "cylinder":
        spawn = sim_utils.CylinderCfg(radius=size / 2.0, height=size, **common)
    else:
        spawn = sim_utils.SphereCfg(radius=size / 2.0, **common)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, size / 2.0)),
    )


@configclass
class DexReMoESceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )
    robot = ALLEGRO_DEXREMOE_CFG


# ---------------------------------------------------------------- actions


@configclass
class ActionsCfg:
    # Grasp-relative action: `action=0` maps to the fixed nominal grasp pose,
    # so a freshly-initialized policy already holds a light-close grasp.
    joint_pos = dex_mdp.GraspRelativeActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.175,
        alpha=0.8,
    )


# ---------------------------------------------------------------- observations


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(func=dex_mdp.policy_obs, history_length=3)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        obs = ObsTerm(func=dex_mdp.privileged_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PointCloudCfg(ObsGroup):
        obs = ObsTerm(func=dex_mdp.point_cloud_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SizeEncodingCfg(ObsGroup):
        obs = ObsTerm(func=dex_mdp.size_encoding_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    point_cloud: PointCloudCfg = PointCloudCfg()
    size_encoding: SizeEncodingCfg = SizeEncodingCfg()


# ---------------------------------------------------------------- commands


@configclass
class CommandsCfg:
    object_pose = dex_mdp.DexReMoECommandCfg(robot_name="robot")


# ---------------------------------------------------------------- rewards


@configclass
class RewardsCfg:
    success = RewTerm(func=dex_mdp.success_bonus, weight=800.0)
    orientation = RewTerm(func=dex_mdp.orientation_inv_track, weight=1.0)
    pos_penalty = RewTerm(func=dex_mdp.position_l2_penalty, weight=-10.0)
    joint_vel = RewTerm(
        func=dex_mdp.joint_vel_clipped_penalty, weight=-0.0002, params={"clip": 6.0}
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


# ---------------------------------------------------------------- terminations


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(func=dex_mdp.object_dropped)
    object_far = DoneTerm(func=dex_mdp.object_far)
    max_success = DoneTerm(func=dex_mdp.max_consecutive_success_reached, params={"cap": 25.0})


# ---------------------------------------------------------------- events


@configclass
class EventCfg:
    reset = EventTerm(
        func=dex_mdp.ResetFromGraspCache,
        mode="reset",
        params={"cache": None, "seed": 0, "eval_seed": 0},
    )


# ---------------------------------------------------------------- env cfg


@configclass
class DexReMoEReorientEnvCfg(ManagerBasedRLEnvCfg):
    scene: DexReMoESceneCfg = DexReMoESceneCfg(num_envs=128, env_spacing=3.0)
    # Use SimulationCfg defaults (friction 0.5/0.5, bounce 0.5) so the training
    # physics matches validate_grasp_stability.py exactly. The earlier explicit
    # 1.0/1.0 friction double-counted contact and made the object stick to the
    # fingers, which showed up as 2cm drift in V2.
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # DexReMoE-specific config
    cache_root: str = "assets/grasp_cache_isaac_validated"
    cache_split: str = "train"
    data_bucket: str | None = None  # None -> stratified 1/3; small/medium/large -> single bucket
    eval_n_per_file: int = 30
    eval_seed: int = 0
    scale_names: tuple = SCALE_NAMES
    scale_values: dict = SCALE_VALUES
    geom_names: tuple = GEOM_NAMES
    hora_base_size_m: float = HORA_BASE_SIZE_M
    point_cloud_points: int = 256
    cache_hand_z: float = 0.5  # hand root z when the cache was validated

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)

        # spawn 33 prims (11 scales x 3 shapes) on the scene
        for s_idx, s_name in enumerate(SCALE_NAMES):
            size = HORA_BASE_SIZE_M * SCALE_VALUES[s_name]
            for g_idx, g_name in enumerate(GEOM_NAMES):
                prim_name = f"obj_{s_idx}_{g_idx}"
                setattr(self.scene, prim_name, _make_obj_cfg(prim_name, size, g_name))
