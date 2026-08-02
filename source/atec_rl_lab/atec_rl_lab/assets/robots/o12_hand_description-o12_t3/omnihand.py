"""Isaac Lab configuration for the OmniHand Pro left hand.

Joint limits and motor limits are read from the URDF. The PD gains stay aligned with
the official OmniHand Pro model so proximal joints, especially the thumb abduction
joint, can hold their pose while downstream joints move.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


OMNIHAND_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "urdf_mesh_col",
    "omnihand_pro_left.urdf",
)

ACTION_SCALE = 1.0
ACTION_OFFSET = 0.0


def _read_urdf_joint_parameters(urdf_path: str) -> tuple[dict[str, float], dict[str, float], set[str]]:
    """Return effort limits, velocity limits and mimic joints from a URDF."""
    root = ET.parse(urdf_path).getroot()
    effort_limits: dict[str, float] = {}
    velocity_limits: dict[str, float] = {}
    mimic_joints: set[str] = set()

    for joint in root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue

        name = joint.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            raise ValueError(f"Actuated joint in {urdf_path!r} is missing its name or limit element")

        effort = limit.get("effort")
        velocity = limit.get("velocity")
        if effort is None or velocity is None:
            raise ValueError(f"Joint {name!r} is missing effort or velocity in {urdf_path!r}")

        effort_limits[name] = float(effort)
        velocity_limits[name] = float(velocity)
        if joint.find("mimic") is not None:
            mimic_joints.add(name)

    return effort_limits, velocity_limits, mimic_joints


OMNIHAND_EFFORT_LIMITS, OMNIHAND_VELOCITY_LIMITS, OMNIHAND_MIMIC_JOINTS = (
    _read_urdf_joint_parameters(OMNIHAND_URDF_PATH)
)
OMNIHAND_JOINT_NAMES = tuple(OMNIHAND_EFFORT_LIMITS)
OMNIHAND_ACTION_JOINT_NAMES = tuple(
    name for name in OMNIHAND_JOINT_NAMES if name not in OMNIHAND_MIMIC_JOINTS
)
OMNIHAND_ACTION_EFFORT_LIMITS = {
    name: OMNIHAND_EFFORT_LIMITS[name] for name in OMNIHAND_ACTION_JOINT_NAMES
}
OMNIHAND_ACTION_VELOCITY_LIMITS = {
    name: OMNIHAND_VELOCITY_LIMITS[name] for name in OMNIHAND_ACTION_JOINT_NAMES
}

# These values match the official OmniHand Pro actuator settings.
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
    spawn=sim_utils.UrdfFileCfg(
        asset_path=OMNIHAND_URDF_PATH,
        force_usd_conversion=True,
        fix_base=False,
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
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
