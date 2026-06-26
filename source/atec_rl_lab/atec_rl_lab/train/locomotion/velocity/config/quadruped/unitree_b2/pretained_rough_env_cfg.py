# Reference: https://github.com/fan-ziqi/robot_lab

import math

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg

from atec_rl_lab.assets.robots import UNITREE_B2_CFG
from isaaclab.sensors import MultiMeshRayCasterCfg
from atec_rl_lab.train.locomotion.velocity.mdp import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains.config.rough2 import ROUGH_TERRAINS_CFG  # isort: skip

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
import atec_rl_lab.tasks.task_d.mdp as task_d_mdp

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.tasks.task_d.terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg
# from atec_rl_lab.tasks.terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg
import copy

# from atec_rl_lab.atec_rl_lab.train.locomotion.velocity.mdp.rewards import track_pos_box_exp

LOW_LEVEL_ENV_CFG = UnitreeB2RoughEnvCfg()
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG
action_low  = LOW_LEVEL_ENV_CFG.actions.joint_pos
# action_low.joint_names = UNITREE_B2_PIPER_CFG.leg_joint_names
action_low.joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
action_low.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}

print(action_low.joint_names,"4444444444")
@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"/home/kh/hkh/code/competition/ATEC2026_Simulation_Challenge/scripts/demo(tq)/policy.pt",
        low_level_decimation=3,
        low_level_actions=action_low,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        clip = (-3,3)
    )
    # For manipulator joint
    # joint_arm = mdp.JointPositionActionCfg(
    #     asset_name="robot", joint_names=[""], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
    # )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel,
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # velocity_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "base_velocity"},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # velocity_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "base_velocity"},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        box_pos = ObsTerm(
            func=mdp.obs_box_rel_robot,
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        
        # joint_effort = ObsTerm(
        #     func=mdp.joint_effort,
        #     clip=(-100, 100),
        #     scale=0.01,
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class ProprioObservationsCfg(ObsGroup):
        """Observations for proprioception group."""
        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        # )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ExteroObservationsCfg(ObsGroup):
        """Observations for exteroception group."""

        # observation terms (order preserved)
        lidar_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # critic: CriticCfg = CriticCfg()
    critic: CriticCfg = CriticCfg()

    extero: ExteroObservationsCfg = ExteroObservationsCfg()


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ATEC_ASSETS_MODEL_DIR}/scene/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ATEC_ASSETS_MODEL_DIR}/scene/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    lidar_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.1,
        pattern_cfg=patterns.LidarPatternCfg(
            vertical_fov_range=(-20.0, 20.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
            channels=16,
        ),
        max_distance=10.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )



@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformThresholdVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.0},
        time_out=True,
    )
    box_terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("box"), "distance_buffer": 0.0},
        time_out=True,
    )

    # Contact sensor
    # illegal_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     # params={"sensor_cfg": SceneEntityCfg("contact_forces", "body_names": ["base_link", ".*_hip"]), "threshold": 1.0},
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names= ["base_link", ".*_hip"]), "threshold": 1.0},

    # )
    robot_tilt_over = DoneTerm(
        func=mdp.bad_orientation,
        params={   "limit_angle": 0.7854,  # 45度
                "asset_cfg": SceneEntityCfg("robot")},
    )




@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # Skip: inertia updated via mass randomization by setting recompute_inertia=True
    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        # func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_with_box3,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


    # randomize_reset_base_box = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #            "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0, 0),
    #             "z": (-0, 0),
    #             "roll": (-0, 0),
    #             "pitch": (-0, 0),
    #             "yaw": (-0, 0),
    #         },
    #         "asset_cfg": SceneEntityCfg("box"),
    #     },
    # )

    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # 箱子移动
    # track_pos_box_exp()
    box_pos = RewTerm(func=mdp.track_box_x_target_exp, weight=15.0)
    box_into_pit_bonus = RewTerm(
        func=task_d_mdp.RewardBoxXInRange,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "x_min": -1.4,
            "x_max": -0.7,
            "reward_value": 20.0,
            "one_time": True,
            "debug": False,
        },
        weight=1.0,
    )
    robot_pos = RewTerm(func=mdp.track_robot_x_when_box_past_minus_05_exp, weight=10.0)
    robot_pos2 = RewTerm(func=mdp.track_robot_close_to_box_before_x_minus05_exp, weight=10.0)
    robot_head_pos = RewTerm(func=mdp.track_robot_head_to_box_before_x_minus05_exp, weight=8.0)
    robot_goal_bonus = RewTerm(
        func=task_d_mdp.RewardCrossX,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": [2.0],
            "reward_value": [30.0],
            "debug": False,
            "visual_assets": False,
        },
        weight=1.0,
    )
    action = RewTerm(func=mdp.action_xy_target_reg, weight=-0.8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)


# 机器人位置



@configclass
class UnitreeB2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()

    terminations: TerminationsCfg = TerminationsCfg()
    
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on
    pit_width_range: tuple[float, float] = (1.3, 1.4)
    platform_height_range: tuple[float, float] = (1.0, 1.2)

    def _build_terrain_cfg(self):
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = self.pit_width_range
            pit_cfg.platform_height_range = self.platform_height_range
        return terrain_cfg

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain = self._build_terrain_cfg()
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 1.0, 0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0, 0, 0.5), # -3, 1.6, 0.5
            ),
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
        #         pos=(0, 0.0, 0.3),
        #     )
        # )
        # joint_names = UNITREE_B2_PIPER_CFG.joint_names
        # leg_joint_names = UNITREE_B2_PIPER_CFG.leg_joint_names
        # arm_joint_names = UNITREE_B2_PIPER_CFG.arm_joint_names
        # print(f"joint_names: {joint_names}")
        # print(f"leg_joint_names: {leg_joint_names}")
        # print(f"arm_joint_names: {arm_joint_names}")





        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        if self.scene.lidar_sensor is not None:
            lidar_sensor = self.scene.lidar_sensor
            self.scene.lidar_sensor = MultiMeshRayCasterCfg(
                prim_path=lidar_sensor.prim_path,
                update_period=lidar_sensor.update_period,
                pattern_cfg=lidar_sensor.pattern_cfg,
                max_distance=lidar_sensor.max_distance,
                debug_vis= True,#lidar_sensor.debug_vis,
                offset=lidar_sensor.offset,
                attach_yaw_only=lidar_sensor.attach_yaw_only,
                ray_alignment=lidar_sensor.ray_alignment,
                drift_range=lidar_sensor.drift_range,
                ray_cast_drift_range=lidar_sensor.ray_cast_drift_range,
                visualizer_cfg=lidar_sensor.visualizer_cfg,
                mesh_prim_paths=[
                    "/World/ground",
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="{ENV_REGEX_NS}/Box",
                        is_shared=True,
                        track_mesh_transforms=True,
                    ),
                ],
            )

        # self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names


        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = joint_names
        # self.observations.policy.height_scan = None
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale

        # self.actions.joint_leg.joint_names = leg_joint_names
        # self.actions.joint_arm.joint_names = arm_joint_names
        # self.actions.joint_wheel = None
        # self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)

        # ------------------------------Rewards------------------------------
        # # General
        # self.rewards.is_terminated.weight = 0

        # # Root penalties
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.1
        # self.rewards.flat_orientation_l2.weight = 0
        # self.rewards.base_height_l2.weight = 0
        # self.rewards.base_height_l2.params["target_height"] = 0.53
        # self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # self.rewards.body_lin_acc_l2.weight = 0
        # self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # # Joint penalties
        # self.rewards.joint_torques_l2.weight = -1e-5
        # self.rewards.joint_vel_l2.weight = 0
        # self.rewards.joint_acc_l2.weight = -1e-7
        # # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        # self.rewards.joint_pos_limits.weight = -5.0
        # self.rewards.joint_vel_limits.weight = 0
        # self.rewards.joint_power.weight = -1e-5
        # self.rewards.stand_still.weight = -2.0
        # self.rewards.joint_pos_penalty.weight = -1.0
        # self.rewards.joint_mirror.weight = -0.1
        # self.rewards.joint_mirror.params["mirror_joints"] = [
        #     ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        #     ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        # ]

        # # Action penalties
        # self.rewards.action_rate_l2.weight = -0.01

        # # Contact sensor
        # self.rewards.undesired_contacts.weight = -1.0
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # self.rewards.contact_forces.weight = -1.5e-4
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # # Velocity-tracking rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # self.rewards.track_ang_vel_z_exp.weight = 3.0

        # # Others
        # self.rewards.feet_air_time.weight = 0
        # self.rewards.feet_air_time.params["threshold"] = 0.5
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_contact.weight = 0
        # self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_contact_without_cmd.weight = 0.1
        # self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_stumble.weight = 0
        # self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_slide.weight = 0
        # self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_height.weight = 0
        # self.rewards.feet_height.params["target_height"] = 0.05
        # self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_height_body.weight = -5.0
        # self.rewards.feet_height_body.params["target_height"] = -0.4
        # self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_gait.weight = 0
        # self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # self.rewards.upward.weight = 3.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum = None
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_lin_vel = None
        # self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2)
        # self.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
