"""Allegro hand config for DexReMoE base-policy env.

Matches the physics used to validate the HORA cache:
  - ImplicitActuator, stiffness=3.0, damping=1.0
  - effort_limit_sim=0.5
  - root link fixed to the env frame
  - hand root raised to z=0.6 to leave clearance above the ground
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG


ALLEGRO_DEXREMOE_CFG = ALLEGRO_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ALLEGRO_HAND_CFG.init_state.replace(pos=(0.0, 0.0, 1.0)),
    spawn=ALLEGRO_HAND_CFG.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=1.0,
            friction=0.01,
        ),
    },
)
