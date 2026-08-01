"""Rough velocity locomotion configuration for OmniHand Pro.

This follows the Unitree B2 velocity task structure, but uses the free-base
OmniHand Pro articulation and hand support links instead of quadruped feet.
The hand is initialized just above the ground so the palm and curled fingers
can generate traction through contact.
"""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from atec_rl_lab.assets.robots.omnihand_pro import OMNIHAND_ACTION_JOINT_NAMES, OMNIHAND_CFG
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class OmniHandProRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Velocity-tracking task for a free-base OmniHand Pro left hand."""

    base_link_name = "base_link"
    base_body_name = "L_palm"
    support_link_name = (
        "L_(palm|thumb_dip_link|index_dip_link|middle_dip_link|ring_dip_link|pinky_dip_link)"
    )
    joint_names = list(OMNIHAND_ACTION_JOINT_NAMES)

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 1.0
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.robot = OMNIHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=OMNIHAND_CFG.init_state.replace(pos=(0.0, 0.0, 0.008)),
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        if self.scene.lidar_sensor is not None:
            lidar_sensor = self.scene.lidar_sensor
            self.scene.lidar_sensor = MultiMeshRayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
                update_period=lidar_sensor.update_period,
                pattern_cfg=lidar_sensor.pattern_cfg,
                max_distance=lidar_sensor.max_distance,
                debug_vis=False,
                offset=lidar_sensor.offset,
                attach_yaw_only=lidar_sensor.attach_yaw_only,
                ray_alignment=lidar_sensor.ray_alignment,
                drift_range=lidar_sensor.drift_range,
                ray_cast_drift_range=lidar_sensor.ray_cast_drift_range,
                visualizer_cfg=lidar_sensor.visualizer_cfg,
                mesh_prim_paths=["/World/ground"],
            )

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = 0.35
        self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}

        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.05

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.01),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-math.pi, math.pi),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_body_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.05, 0.05)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_body_name]
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.005, 0.005),
        }
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_body_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-0.1, 0.1)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_body_name]
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_body_name]

        self.rewards.joint_torques_l2.weight = -1.0e-4
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_vel_l2.weight = -1.0e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_acc_l2.weight = -1.0e-6
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_vel_limits.weight = -0.1
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_power.weight = -1.0e-4
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.weight = -0.2
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.weight = -0.05
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names

        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.support_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.04)
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.05
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.support_link_name]
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.upward.weight = 1.0

        self.terminations.illegal_contact = None
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)

        if self.__class__.__name__ == "OmniHandProRoughEnvCfg":
            self.disable_zero_weight_rewards()
