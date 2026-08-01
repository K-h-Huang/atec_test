"""Isaac Lab configuration for the OmniHand Pro left hand.

Joint limits are read directly from the URDF.  The actuator parameters use the
heuristic described in the supplied reference:

    kp = I * omega**2
    kd = 2 * I * zeta * omega
    action_scale = 0.25 * effort_limit / kp

With action_scale=1, natural frequency=10 Hz and damping ratio=2, these
relations determine stiffness, damping and armature from each URDF effort
limit.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


OMNIHAND_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "urdf",
    "omnihand_pro_left.urdf",
)

ACTION_SCALE = 1.0
ACTION_OFFSET = 0.0
NATURAL_FREQUENCY_HZ = 10.0
DAMPING_RATIO = 2.0


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

# alpha = 0.25 * tau_max / kp, with alpha (action scale) fixed to 1.
OMNIHAND_STIFFNESS = {
    name: 0.25 * OMNIHAND_EFFORT_LIMITS[name] / ACTION_SCALE
    for name in OMNIHAND_ACTION_JOINT_NAMES
}

# omega is angular natural frequency in rad/s.
_OMEGA = 2.0 * math.pi * NATURAL_FREQUENCY_HZ
OMNIHAND_ARMATURE = {
    name: stiffness / (_OMEGA**2)
    for name, stiffness in OMNIHAND_STIFFNESS.items()
}
OMNIHAND_DAMPING = {
    name: 2.0 * OMNIHAND_ARMATURE[name] * DAMPING_RATIO * _OMEGA
    for name in OMNIHAND_ACTION_JOINT_NAMES
}


OMNIHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=OMNIHAND_URDF_PATH,
        fix_base=True,
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
            armature=OMNIHAND_ARMATURE,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        )
    },
)


# Per-action dictionaries only include independent (non-mimic) joints.
OMNIHAND_ACTION_SCALE = {name: ACTION_SCALE for name in OMNIHAND_ACTION_JOINT_NAMES}
OMNIHAND_ACTION_OFFSET = {name: ACTION_OFFSET for name in OMNIHAND_ACTION_JOINT_NAMES}
