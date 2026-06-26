from __future__ import annotations

import copy
import os
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.assets.objects import Banana_cfg, Mustard_cfg, Sugar_cfg
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG
from atec_rl_lab.tasks.task_b.terrain import TASK_B_TERRAIN_CFG
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import (
    UnitreeB2RoughEnvCfg as LowLevelUnitreeB2RoughEnvCfg,
)
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


NUM_TASK_OBJECTS = 18
TARGET_CENTER = (7.0, 0.0)
TARGET_RADIUS = 1.0
TARGET_Z_RANGE = (0.0, 0.5)
ROBOT_SPAWN_X = -7.0
ROBOT_SPAWN_Y = 0.0
OBJECT_SPAWN_X_RANGE = (-7.5, -1.5)
OBJECT_SPAWN_Y_RANGE = (-2.5, 2.5)
MIN_OBJECT_ROBOT_DISTANCE = 0.9
MIN_OBJECT_OBJECT_DISTANCE = 0.4
TERRAIN_HALF_EXTENT = 9.5
LOW_LEVEL_POLICY_PATH = os.getenv(
    "ATEC_B2_LOCO_POLICY",
    "/home/kh/hkh/code/competition/ATEC2026_Simulation_Challenge/scripts/demo(tq)/policy.pt",
)


LOW_LEVEL_ENV_CFG = LowLevelUnitreeB2RoughEnvCfg()
LOW_LEVEL_ACTION_CFG = copy.deepcopy(LOW_LEVEL_ENV_CFG.actions.joint_pos)
LOW_LEVEL_ACTION_CFG.joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
LOW_LEVEL_ACTION_CFG.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
LOW_LEVEL_OBS_CFG = copy.deepcopy(LOW_LEVEL_ENV_CFG.observations.policy)
LOW_LEVEL_OBS_CFG.joint_pos.params["asset_cfg"].joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
LOW_LEVEL_OBS_CFG.joint_vel.params["asset_cfg"].joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)


def _object_name(index: int) -> str:
    return f"object_{index}"


def _get_ee_body_idx(env: ManagerBasedEnv, ee_body_name: str = "gripper_base") -> int:
    cache_name = "_task_b_ee_body_idx"
    body_idx = getattr(env, cache_name, None)
    if body_idx is None:
        robot = env.scene["robot"]
        body_ids, found_names = robot.find_bodies(ee_body_name)
        if len(body_ids) == 0:
            raise ValueError(f"Cannot find end-effector body '{ee_body_name}'.")
        body_idx = int(body_ids[0])
        setattr(env, cache_name, body_idx)
        print(f"[taskb] using end-effector body: {found_names[0]} (idx={body_idx})")
    return body_idx


def _get_object_positions_local(env: ManagerBasedEnv) -> torch.Tensor:
    env_origins = env.scene.env_origins[:, None, :3]
    object_positions = []
    for object_idx in range(1, NUM_TASK_OBJECTS + 1):
        obj = env.scene[_object_name(object_idx)]
        object_positions.append(obj.data.root_pos_w[:, :3] - env_origins[:, 0, :])
    return torch.stack(object_positions, dim=1)


def _get_robot_root_local(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def _get_ee_local(env: ManagerBasedEnv, ee_body_name: str = "gripper_base") -> torch.Tensor:
    robot = env.scene["robot"]
    ee_body_idx = _get_ee_body_idx(env, ee_body_name)
    return robot.data.body_pos_w[:, ee_body_idx, :3] - env.scene.env_origins[:, :3]


def _objects_in_target_mask(
    object_positions_local: torch.Tensor,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
    z_min: float = TARGET_Z_RANGE[0],
    z_max: float = TARGET_Z_RANGE[1],
) -> torch.Tensor:
    center_xy = torch.tensor(center, device=object_positions_local.device, dtype=torch.float32)
    dist_sq = torch.sum((object_positions_local[:, :, :2] - center_xy) ** 2, dim=-1)
    inside_xy = dist_sq <= float(radius) * float(radius)
    inside_z = (object_positions_local[:, :, 2] >= float(z_min)) & (object_positions_local[:, :, 2] <= float(z_max))
    return inside_xy & inside_z


def _select_nearest_active_object(
    object_positions_local: torch.Tensor,
    reference_positions_local: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_mask is None:
        active_mask = torch.ones(
            (object_positions_local.shape[0], object_positions_local.shape[1]),
            device=object_positions_local.device,
            dtype=torch.bool,
        )

    dist_xy = torch.linalg.norm(object_positions_local[:, :, :2] - reference_positions_local[:, None, :2], dim=-1)
    dist_xy = dist_xy.masked_fill(~active_mask, float("inf"))
    nearest_indices = torch.argmin(dist_xy, dim=1)
    batch_indices = torch.arange(object_positions_local.shape[0], device=object_positions_local.device)
    nearest_positions = object_positions_local[batch_indices, nearest_indices]
    has_active = active_mask.any(dim=1)
    nearest_positions = torch.where(has_active[:, None], nearest_positions, torch.zeros_like(nearest_positions))
    return nearest_positions, has_active


def obs_target_rel_robot(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
) -> torch.Tensor:
    robot = env.scene["robot"]
    robot_root_local = _get_robot_root_local(env)
    target_local = torch.tensor(
        [center[0], center[1], 0.0],
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    rel_pos_local = target_local - robot_root_local
    rel_pos_body = math_utils.quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w), rel_pos_local)
    return rel_pos_body[:, :2]


def obs_nearest_active_object_rel_robot(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
) -> torch.Tensor:
    robot = env.scene["robot"]
    object_positions_local = _get_object_positions_local(env)
    active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
    robot_root_local = _get_robot_root_local(env)
    nearest_object_local, has_active = _select_nearest_active_object(object_positions_local, robot_root_local, active_mask)
    rel_pos_local = nearest_object_local - robot_root_local
    rel_pos_body = math_utils.quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w), rel_pos_local)
    return torch.where(has_active[:, None], rel_pos_body, torch.zeros_like(rel_pos_body))


def obs_nearest_active_object_rel_ee(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
    ee_body_name: str = "gripper_base",
) -> torch.Tensor:
    robot = env.scene["robot"]
    object_positions_local = _get_object_positions_local(env)
    active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
    ee_local = _get_ee_local(env, ee_body_name=ee_body_name)
    nearest_object_local, has_active = _select_nearest_active_object(object_positions_local, ee_local, active_mask)
    rel_pos_local = nearest_object_local - ee_local
    rel_pos_body = math_utils.quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w), rel_pos_local)
    return torch.where(has_active[:, None], rel_pos_body, torch.zeros_like(rel_pos_body))


def obs_placed_object_ratio(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
) -> torch.Tensor:
    object_positions_local = _get_object_positions_local(env)
    inside_target = _objects_in_target_mask(object_positions_local, center=center, radius=radius)
    return inside_target.to(torch.float32).mean(dim=1, keepdim=True)


def reward_robot_to_active_object_exp(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
    std: float = 2.0,
) -> torch.Tensor:
    object_positions_local = _get_object_positions_local(env)
    active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
    robot_root_local = _get_robot_root_local(env)
    nearest_object_local, has_active = _select_nearest_active_object(object_positions_local, robot_root_local, active_mask)
    dist = torch.linalg.norm(nearest_object_local[:, :2] - robot_root_local[:, :2], dim=1)
    reward = torch.exp(-(dist * dist) / (2.0 * float(std) * float(std)))
    return reward * has_active.to(torch.float32)


def reward_ee_to_active_object_exp(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
    ee_body_name: str = "gripper_base",
    std: float = 0.45,
) -> torch.Tensor:
    object_positions_local = _get_object_positions_local(env)
    active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
    ee_local = _get_ee_local(env, ee_body_name=ee_body_name)
    nearest_object_local, has_active = _select_nearest_active_object(object_positions_local, ee_local, active_mask)
    dist = torch.linalg.norm(nearest_object_local - ee_local, dim=1)
    reward = torch.exp(-(dist * dist) / (2.0 * float(std) * float(std)))
    return reward * has_active.to(torch.float32)


def reward_active_object_to_target_exp(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
    std: float = 3.0,
) -> torch.Tensor:
    object_positions_local = _get_object_positions_local(env)
    active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
    target_local = torch.tensor(center, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
    dist_to_target = torch.linalg.norm(object_positions_local[:, :, :2] - target_local[:, None, :], dim=-1)
    dist_to_target = dist_to_target.masked_fill(~active_mask, float("inf"))
    nearest_dist = torch.min(dist_to_target, dim=1).values
    has_active = active_mask.any(dim=1)
    nearest_dist = torch.where(has_active, nearest_dist, torch.zeros_like(nearest_dist))
    reward = torch.exp(-(nearest_dist * nearest_dist) / (2.0 * float(std) * float(std)))
    return reward * has_active.to(torch.float32)


def reward_active_object_lift(
    env: ManagerBasedEnv,
    center: tuple[float, float] = TARGET_CENTER,
    radius: float = TARGET_RADIUS,
    min_height: float = 0.12,
    target_height: float = 0.35,
) -> torch.Tensor:
    object_positions_local = _get_object_positions_local(env)
    active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
    target_local = torch.tensor(center, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
    dist_to_target = torch.linalg.norm(object_positions_local[:, :, :2] - target_local[:, None, :], dim=-1)
    dist_to_target = dist_to_target.masked_fill(~active_mask, float("inf"))
    nearest_indices = torch.argmin(dist_to_target, dim=1)
    batch_indices = torch.arange(object_positions_local.shape[0], device=object_positions_local.device)
    nearest_object_local = object_positions_local[batch_indices, nearest_indices]
    has_active = active_mask.any(dim=1)
    lift = (nearest_object_local[:, 2] - float(min_height)) / max(float(target_height - min_height), 1.0e-6)
    lift = torch.clamp(lift, 0.0, 1.0)
    return lift * has_active.to(torch.float32)


class ObjectsInTargetLocal(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._counted = torch.zeros((env.num_envs, NUM_TASK_OBJECTS), device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._counted.fill_(False)
        else:
            self._counted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        center: tuple[float, float] = TARGET_CENTER,
        radius: float = TARGET_RADIUS,
        reward_per_object: float = 8.0,
        z_min: float = TARGET_Z_RANGE[0],
        z_max: float = TARGET_Z_RANGE[1],
    ) -> torch.Tensor:
        object_positions_local = _get_object_positions_local(env)
        inside_target = _objects_in_target_mask(
            object_positions_local,
            center=center,
            radius=radius,
            z_min=z_min,
            z_max=z_max,
        )
        newly_inside = inside_target & (~self._counted)
        self._counted |= inside_target
        return newly_inside.sum(dim=1).to(torch.float32) * float(reward_per_object)


class GraspedObjectsByEELocal(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._counted = torch.zeros((env.num_envs, NUM_TASK_OBJECTS), device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._counted.fill_(False)
        else:
            self._counted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        ee_body_name: str = "gripper_base",
        grasp_dist_thresh: float = 0.20,
        min_lift: float = 0.12,
        reward_per_object: float = 3.0,
        center: tuple[float, float] = TARGET_CENTER,
        radius: float = TARGET_RADIUS,
    ) -> torch.Tensor:
        object_positions_local = _get_object_positions_local(env)
        active_mask = ~_objects_in_target_mask(object_positions_local, center=center, radius=radius)
        ee_local = _get_ee_local(env, ee_body_name=ee_body_name)
        dist = torch.linalg.norm(object_positions_local - ee_local[:, None, :], dim=-1)
        lifted = object_positions_local[:, :, 2] >= float(min_lift)
        grasped = (dist <= float(grasp_dist_thresh)) & lifted & active_mask
        newly_grasped = grasped & (~self._counted)
        self._counted |= grasped
        return newly_grasped.sum(dim=1).to(torch.float32) * float(reward_per_object)


class AllObjectsInTargetLocal(ManagerTermBase):
    def __call__(
        self,
        env: ManagerBasedEnv,
        center: tuple[float, float] = TARGET_CENTER,
        radius: float = TARGET_RADIUS,
        z_min: float = TARGET_Z_RANGE[0],
        z_max: float = TARGET_Z_RANGE[1],
    ) -> torch.Tensor:
        object_positions_local = _get_object_positions_local(env)
        inside_target = _objects_in_target_mask(
            object_positions_local,
            center=center,
            radius=radius,
            z_min=z_min,
            z_max=z_max,
        )
        return inside_target.all(dim=1)


class AnyObjectOutOfBoundsLocal(ManagerTermBase):
    def __call__(
        self,
        env: ManagerBasedEnv,
        xy_limit: float = TERRAIN_HALF_EXTENT,
        z_min: float = -0.2,
    ) -> torch.Tensor:
        object_positions_local = _get_object_positions_local(env)
        out_of_xy = (object_positions_local[:, :, 0].abs() > float(xy_limit)) | (
            object_positions_local[:, :, 1].abs() > float(xy_limit)
        )
        out_of_z = object_positions_local[:, :, 2] < float(z_min)
        return (out_of_xy | out_of_z).any(dim=1)


def reset_task_b_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = OBJECT_SPAWN_X_RANGE,
    y_range: tuple[float, float] = OBJECT_SPAWN_Y_RANGE,
    target_center: tuple[float, float] = TARGET_CENTER,
    target_radius: float = TARGET_RADIUS,
    min_robot_distance: float = MIN_OBJECT_ROBOT_DISTANCE,
    min_object_distance: float = MIN_OBJECT_OBJECT_DISTANCE,
):
    device = env.device
    env_origins = env.scene.env_origins[env_ids, :3]
    robot_root_local = torch.tensor(
        [ROBOT_SPAWN_X, ROBOT_SPAWN_Y, 0.0],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(len(env_ids), 1)

    placed_xy: list[torch.Tensor] = []
    for object_idx in range(1, NUM_TASK_OBJECTS + 1):
        obj = env.scene[_object_name(object_idx)]
        root_states = obj.data.default_root_state[env_ids].clone()

        sampled_xy = torch.zeros((len(env_ids), 2), device=device, dtype=torch.float32)
        valid_mask = torch.zeros(len(env_ids), device=device, dtype=torch.bool)

        for _ in range(16):
            candidate_xy = torch.zeros_like(sampled_xy)
            candidate_xy[:, 0] = math_utils.sample_uniform(
                float(x_range[0]), float(x_range[1]), (len(env_ids),), device=device
            )
            candidate_xy[:, 1] = math_utils.sample_uniform(
                float(y_range[0]), float(y_range[1]), (len(env_ids),), device=device
            )

            dist_robot = torch.linalg.norm(candidate_xy - robot_root_local[:, :2], dim=1)
            dist_target = torch.linalg.norm(
                candidate_xy
                - torch.tensor(target_center, device=device, dtype=torch.float32).unsqueeze(0),
                dim=1,
            )
            candidate_valid = (dist_robot >= float(min_robot_distance)) & (
                dist_target >= float(target_radius + 0.35)
            )

            if placed_xy:
                stacked_xy = torch.stack(placed_xy, dim=1)
                dist_prev = torch.linalg.norm(stacked_xy - candidate_xy[:, None, :], dim=-1)
                candidate_valid &= (dist_prev >= float(min_object_distance)).all(dim=1)

            write_mask = (~valid_mask) & candidate_valid
            sampled_xy[write_mask] = candidate_xy[write_mask]
            valid_mask |= candidate_valid

            if bool(valid_mask.all()):
                break

        if not bool(valid_mask.all()):
            fallback_xy = robot_root_local[:, :2] + torch.tensor([2.0, 0.0], device=device, dtype=torch.float32)
            sampled_xy[~valid_mask] = fallback_xy[~valid_mask]

        placed_xy.append(sampled_xy)

        root_states[:, 0] = sampled_xy[:, 0] + env_origins[:, 0]
        root_states[:, 1] = sampled_xy[:, 1] + env_origins[:, 1]
        root_states[:, 2] = root_states[:, 2] + env_origins[:, 2]
        root_states[:, 7:13] = 0.0

        obj.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
        obj.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)


def _build_default_object_cfg(index: int):
    row = (index - 1) // 6
    col = (index - 1) % 6
    x = OBJECT_SPAWN_X_RANGE[0] + 0.8 * col
    y = OBJECT_SPAWN_Y_RANGE[0] + 1.8 * row
    if index <= 6:
        return Sugar_cfg([x, y, 0.15], [0.0, 0.707, 0.0, 0.707], f"Object{index}")
    if index <= 12:
        return Mustard_cfg([x, y, 0.10], [0.0, 0.0, -0.707, 0.707], f"Object{index}")
    return Banana_cfg([x, y, 0.10], [0.0, 0.0, -0.707, 0.707], f"Object{index}")


@configclass
class ActionsCfg:
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=3,
        low_level_actions=LOW_LEVEL_ACTION_CFG,
        low_level_observations=LOW_LEVEL_OBS_CFG,
        clip=(-3.0, 3.0),
    )
    joint_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UNITREE_B2_PIPER_CFG.arm_joint_names),
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
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
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        lidar_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
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
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        target_rel_robot = ObsTerm(func=obs_target_rel_robot, clip=(-20.0, 20.0), scale=1.0)
        nearest_object_rel_robot = ObsTerm(
            func=obs_nearest_active_object_rel_robot,
            clip=(-20.0, 20.0),
            scale=1.0,
        )
        nearest_object_rel_ee = ObsTerm(
            func=obs_nearest_active_object_rel_ee,
            clip=(-20.0, 20.0),
            scale=1.0,
        )
        placed_ratio = ObsTerm(func=obs_placed_object_ratio, clip=(0.0, 1.0), scale=1.0)
        lidar_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = copy.deepcopy(TASK_B_TERRAIN_CFG)
    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
    pass


@configclass
class EventCfg:
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.6, 0.6),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-1.2, 1.2),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_objects = EventTerm(
        func=reset_task_b_objects,
        mode="reset",
        params={},
    )


@configclass
class RewardsCfg:
    objects_in_target = RewTerm(
        func=ObjectsInTargetLocal,
        params={
            "center": TARGET_CENTER,
            "radius": TARGET_RADIUS,
            "reward_per_object": 12.0,
        },
        weight=1.0,
    )
    grasped_objects = RewTerm(
        func=GraspedObjectsByEELocal,
        params={
            "ee_body_name": "gripper_base",
            "grasp_dist_thresh": 0.20,
            "min_lift": 0.12,
            "reward_per_object": 5.0,
            "center": TARGET_CENTER,
            "radius": TARGET_RADIUS,
        },
        weight=1.0,
    )
    robot_to_active_object = RewTerm(
        func=reward_robot_to_active_object_exp,
        params={"center": TARGET_CENTER, "radius": TARGET_RADIUS, "std": 2.0},
        weight=0.5,
    )
    ee_to_active_object = RewTerm(
        func=reward_ee_to_active_object_exp,
        params={
            "center": TARGET_CENTER,
            "radius": TARGET_RADIUS,
            "ee_body_name": "gripper_base",
            "std": 0.45,
        },
        weight=2.5,
    )
    active_object_to_target = RewTerm(
        func=reward_active_object_to_target_exp,
        params={"center": TARGET_CENTER, "radius": TARGET_RADIUS, "std": 3.0},
        weight=3.0,
    )
    active_object_lift = RewTerm(
        func=reward_active_object_lift,
        params={"center": TARGET_CENTER, "radius": TARGET_RADIUS, "min_height": 0.12, "target_height": 0.35},
        weight=2.0,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(UNITREE_B2_PIPER_CFG.arm_joint_names))},
        weight=-5e-5,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.0},
        time_out=True,
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[UNITREE_B2_PIPER_CFG.base_link_name, ".*_hip", ".*_thigh"],
            ),
            "threshold": 1.0,
        },
    )
    fall = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.25},
        time_out=False,
    )
    objects_out_of_bounds = DoneTerm(func=AnyObjectOutOfBoundsLocal, params={})
    all_objects_in_target = DoneTerm(
        func=AllObjectsInTargetLocal,
        params={"center": TARGET_CENTER, "radius": TARGET_RADIUS},
        time_out=False,
    )


@configclass
class UnitreeB2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=256, env_spacing=25.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    base_link_name = UNITREE_B2_PIPER_CFG.base_link_name

    def __post_init__(self):
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=(ROBOT_SPAWN_X, ROBOT_SPAWN_Y, 0.68),
            ),
        )

        for object_idx in range(1, NUM_TASK_OBJECTS + 1):
            setattr(self.scene, _object_name(object_idx), _build_default_object_cfg(object_idx))

        super().__post_init__()

        self.episode_length_s = 60.0
        self.scene.terrain = copy.deepcopy(TASK_B_TERRAIN_CFG)
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        joint_names = list(UNITREE_B2_PIPER_CFG.joint_names)
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_arm.joint_names = list(UNITREE_B2_PIPER_CFG.arm_joint_names)
        self.actions.pre_trained_policy_action.low_level_actions.joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
        self.actions.pre_trained_policy_action.low_level_observations.joint_pos.params["asset_cfg"].joint_names = list(
            UNITREE_B2_PIPER_CFG.leg_joint_names
        )
        self.actions.pre_trained_policy_action.low_level_observations.joint_vel.params["asset_cfg"].joint_names = list(
            UNITREE_B2_PIPER_CFG.leg_joint_names
        )

        if self.scene.lidar_sensor is not None:
            lidar_sensor = self.scene.lidar_sensor
            object_targets = [
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=f"{{ENV_REGEX_NS}}/Object{object_idx}",
                    is_shared=True,
                    track_mesh_transforms=True,
                )
                for object_idx in range(1, NUM_TASK_OBJECTS + 1)
            ]
            self.scene.lidar_sensor = MultiMeshRayCasterCfg(
                prim_path=lidar_sensor.prim_path,
                update_period=lidar_sensor.update_period,
                pattern_cfg=lidar_sensor.pattern_cfg,
                max_distance=lidar_sensor.max_distance,
                debug_vis=lidar_sensor.debug_vis,
                offset=lidar_sensor.offset,
                attach_yaw_only=lidar_sensor.attach_yaw_only,
                ray_alignment=lidar_sensor.ray_alignment,
                drift_range=lidar_sensor.drift_range,
                ray_cast_drift_range=lidar_sensor.ray_cast_drift_range,
                visualizer_cfg=lidar_sensor.visualizer_cfg,
                mesh_prim_paths=["/World/ground", *object_targets],
            )
