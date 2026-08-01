"""Isaac Lab configuration for the OmniHand Pro left hand.

Joint limits and motor limits are defined from the official MJCF model.  The PD gains
match the official OmniHand Pro MuJoCo model so that proximal joints, especially
the thumb abduction joint, can hold their pose while downstream joints move.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


OMNIHAND_MJCF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "MJCF",
    "omnihand_pro_left_isaaclab.xml",
)

ACTION_SCALE = 1.0
ACTION_OFFSET = 0.0


OMNIHAND_JOINT_NAMES = (
    "L_thumb_roll_joint",
    "L_thumb_abad_joint",
    "L_thumb_mcp_joint",
    "L_thumb_pip_joint",
    "L_thumb_dip_joint",
    "L_index_abad_joint",
    "L_index_mcp_joint",
    "L_index_pip_joint",
    "L_index_dip_joint",
    "L_middle_abad_joint",
    "L_middle_mcp_joint",
    "L_middle_pip_joint",
    "L_middle_dip_joint",
    "L_ring_mcp_joint",
    "L_ring_pip_joint",
    "L_ring_dip_joint",
    "L_pinky_mcp_joint",
    "L_pinky_pip_joint",
    "L_pinky_dip_joint",
)

# MJCF equality constraints drive these passive joints indirectly.
OMNIHAND_ACTION_JOINT_NAMES = (
    "L_thumb_roll_joint",
    "L_thumb_abad_joint",
    "L_thumb_mcp_joint",
    "L_thumb_pip_joint",
    "L_index_abad_joint",
    "L_index_mcp_joint",
    "L_index_pip_joint",
    "L_middle_abad_joint",
    "L_middle_mcp_joint",
    "L_middle_pip_joint",
    "L_ring_mcp_joint",
    "L_pinky_mcp_joint",
)

OMNIHAND_ACTION_EFFORT_LIMITS = {
    "L_thumb_roll_joint": 0.134,
    "L_thumb_abad_joint": 0.317,
    "L_thumb_mcp_joint": 0.8,
    "L_thumb_pip_joint": 0.92,
    "L_index_abad_joint": 1.0,
    "L_index_mcp_joint": 0.42,
    "L_index_pip_joint": 0.216,
    "L_middle_abad_joint": 1.0,
    "L_middle_mcp_joint": 0.413,
    "L_middle_pip_joint": 0.136,
    "L_ring_mcp_joint": 0.444,
    "L_pinky_mcp_joint": 0.444,
}

OMNIHAND_ACTION_VELOCITY_LIMITS = {
    "L_thumb_roll_joint": 2.38,
    "L_thumb_abad_joint": 2.33,
    "L_thumb_mcp_joint": 1.35,
    "L_thumb_pip_joint": 1.87,
    "L_index_abad_joint": 2.49,
    "L_index_mcp_joint": 2.49,
    "L_index_pip_joint": 2.49,
    "L_middle_abad_joint": 2.49,
    "L_middle_mcp_joint": 2.49,
    "L_middle_pip_joint": 2.49,
    "L_ring_mcp_joint": 2.62,
    "L_pinky_mcp_joint": 2.62,
}
# These values match the official assets/MJCF/omnihand_pro_left.xml actuators.
OMNIHAND_STIFFNESS = {
    "L_thumb_roll_joint": 100.0,
    "L_thumb_abad_joint": 100.0,
    "L_thumb_mcp_joint": 100.0,
    "L_thumb_pip_joint": 100.0,
    "L_index_abad_joint": 100.0,
    "L_index_mcp_joint": 100.0,
    "L_index_pip_joint": 150.0,
    "L_middle_abad_joint": 100.0,
    "L_middle_mcp_joint": 100.0,
    "L_middle_pip_joint": 150.0,
    "L_ring_mcp_joint": 150.0,
    "L_pinky_mcp_joint": 150.0,
}
OMNIHAND_DAMPING = {
    "L_thumb_roll_joint": 0.2,
    "L_thumb_abad_joint": 0.2,
    "L_thumb_mcp_joint": 0.2,
    "L_thumb_pip_joint": 0.1,
    "L_index_abad_joint": 0.2,
    "L_index_mcp_joint": 0.2,
    "L_index_pip_joint": 0.15,
    "L_middle_abad_joint": 0.2,
    "L_middle_mcp_joint": 0.2,
    "L_middle_pip_joint": 0.15,
    "L_ring_mcp_joint": 0.15,
    "L_pinky_mcp_joint": 0.15,
}


OMNIHAND_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=OMNIHAND_MJCF_PATH,
        fix_base=False,
        import_inertia_tensor=True,
        import_sites=True,
        self_collision=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.1, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "hand": ImplicitActuatorCfg(
            joint_names_expr=list(OMNIHAND_ACTION_JOINT_NAMES),
            effort_limit_sim=OMNIHAND_ACTION_EFFORT_LIMITS,
            velocity_limit_sim=OMNIHAND_ACTION_VELOCITY_LIMITS,
            stiffness=OMNIHAND_STIFFNESS,
            damping=OMNIHAND_DAMPING,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        )
    },
)


# Per-action dictionaries only include independent (non-mimic) joints.
OMNIHAND_ACTION_SCALE = {name: ACTION_SCALE for name in OMNIHAND_ACTION_JOINT_NAMES}
OMNIHAND_ACTION_OFFSET = {name: ACTION_OFFSET for name in OMNIHAND_ACTION_JOINT_NAMES}
