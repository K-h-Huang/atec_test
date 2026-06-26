# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward



def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# def track_box_x_target_exp(
#     env: ManagerBasedRLEnv, 
#     std: float = 1.2, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("box")
# ) -> torch.Tensor:
#     """
#     Reward for pushing box toward x=-0.9, y limited to [0.6, 3.4].
#     Big penalty when box x < -1.0.
#     Higher reward the closer box x is to -0.9.
#     """
#     # 提取箱子物体，类型提示对齐原代码
#     box: RigidObject = env.scene[asset_cfg.name]
#     box_pos_w = box.data.root_pos_w  # [num_envs, 3] world坐标
    
#     target_x = 0.0
#     box_x = box_pos_w[:, 0]
#     box_y = box_pos_w[:, 1]

#     # 1. X方向误差：距离目标-0.9的平方误差
#     # print(box_x,"===============")
#     # print(box_y,"==============")
#     x_error_sq = torch.square(box_x - target_x)
#     reward_base = torch.exp(-x_error_sq / (std ** 2))

#     # 2. Y轴约束掩码：仅y ∈ [0.6, 3.4]保留奖励，外部乘衰减系数
#     y_in_range_mask = (box_y <= 0.1) & (box_y >= -3.4)
#     y_scale = torch.where(y_in_range_mask, torch.ones_like(reward_base), torch.full_like(reward_base, 0.05))
#     reward_base *= y_scale

#     # 3. X < -1.0 大幅惩罚，奖励断崖式降低
#     x_over_left_mask = box_x < -1.0
#     x_left_penalty = torch.where(x_over_left_mask, torch.full_like(reward_base, 0.5), torch.ones_like(reward_base))
#     reward = reward_base * x_left_penalty
#     # print(reward, x_over_left_mask)
#     return reward

def action_xy_target_reg(env: ManagerBasedRLEnv, target: float = 1.5) -> torch.Tensor:
    """
    Penalize XY action deviate from target magnitude 1.5,
    discourage too small / too large xy command output.
    Only regularize first two action dims (x,y).
    """
    act = env.action_manager.action
    # 只取xy两维动作
    act_xy = act[:, :2]
    # xy动作平方和
    xy_sq_sum = torch.sum(torch.abs(act_xy), dim=1)
    # 偏离目标幅值1.5的平方损失，过小过大都会产生惩罚
    loss = torch.abs(xy_sq_sum - target ** 2)
    return loss


def _target_xy_like(reference_xy: torch.Tensor, target_x: float, target_y: float) -> torch.Tensor:
    """Create a batched XY target tensor on the same device and dtype as the reference."""
    return reference_xy.new_tensor([target_x, target_y]).unsqueeze(0).expand(reference_xy.shape[0], -1)


def _safe_normalize_xy(vec_xy: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Normalize XY vectors while avoiding division by zero."""
    norm = torch.linalg.norm(vec_xy, dim=1, keepdim=True).clamp_min(eps)
    return vec_xy / norm


def track_box_x_target_exp(
    env: ManagerBasedRLEnv, 
    std: float = 1.8, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box")
) -> torch.Tensor:
    """
    Reward for pushing box toward x=0.0, y soft range [-3.4, 0.1] (smooth decay to avoid sparse reward).
    Mild penalty when box x < -1.0.
    Higher reward the closer box x is to 0.0.
    """
    box: RigidObject = env.scene[asset_cfg.name]
    box_pos_w = box.data.root_pos_w  # [num_envs, 3] world坐标
    # print(box_pos_w[:,2],"z")
    box_xy = box_pos_w[:, :2]
    target_xy = _target_xy_like(box_xy, target_x=0.0, target_y=-1.0)
    diff_xy = box_xy - target_xy
    dist_sq = torch.square(diff_xy[:, 0]) + 1.5 * torch.square(diff_xy[:, 1])
    return torch.exp(-dist_sq / (std**2))

    target_x = 0.0
    target_y = -1.0
    box_x = box_pos_w[:, 0]
    box_y = box_pos_w[:, 1]

    # 1. X方向指数跟踪，std放大到1.8，远距离保留梯度
    x_error_sq = torch.square(box_x - target_x)
    reward_base_x = torch.exp(-x_error_sq / (std ** 2))

    y_error_sq = torch.square(box_y - target_y)
    reward_base_y = torch.exp(-y_error_sq / (std ** 2))
    reward_base = reward_base_x + reward_base_y
    # 2. Y轴平滑衰减，替代硬掩码断崖惩罚（核心解决稀疏）
    y_upper_bound = 0.1
    y_lower_bound = -3.4
    y_out_upper = torch.clamp(box_y - y_upper_bound, min=0.0)  # y超过0.1的偏移量
    y_out_lower = torch.clamp(y_lower_bound - box_y, min=0.0)  # y低于-3.4的偏移量
    y_out_dist = torch.maximum(y_out_upper, y_out_lower)
    # 随y偏离距离指数衰减，不会直接砍死，偏离越多缓慢降低
    y_scale = torch.exp(-y_out_dist / 2.0)
    reward_base *= y_scale

    # 3. X < -1.0 轻度惩罚，从0.5上调到0.7，不要过度压制初始左侧箱子
    x_over_left_mask = box_x < -1.0
    x_left_penalty = torch.where(x_over_left_mask, torch.full_like(reward_base, 0.7), torch.ones_like(reward_base))
    reward = reward_base * x_left_penalty

    return reward

def track_robot_x_when_box_past_minus_05_exp(
    env: ManagerBasedRLEnv, 
    std: float=1.5, 
    box_asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward only when box x > -0.5: robot gets higher reward the closer its x is to 2.2.
    If box x <= -0.5, reward is 0.
    Higher reward the closer robot x is to target 2.2 after box passes x=-0.5.
    """
    # 获取箱子与机器人物体
    box: RigidObject = env.scene[box_asset_cfg.name]
    robot: RigidObject = env.scene[robot_asset_cfg.name]

    box_z = box.data.root_pos_w[:, 2]
    robot_x = robot.data.root_pos_w[:, 0]
    robot_target_x = 2.2

    # 1. 机器人距离目标2.2的指数跟踪基础奖励
    x_error_sq = torch.square(robot_x - robot_target_x)
    reward_base = torch.exp(-x_error_sq / (std ** 2))

    # 2. 箱子z < 0才生效奖励，否则直接置0
    box_pass_mask = box_z < 0
    box_scale = torch.where(box_pass_mask, torch.ones_like(reward_base), torch.zeros_like(reward_base))

    # 最终奖励
    reward = reward_base * box_scale

    return reward


def track_pos_box_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), box_cfg: SceneEntityCfg = SceneEntityCfg("box")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    pos = asset.data.root_pos_w[:, :3]
    box =  env.scene[box_cfg.name]
    pos_box = box.data.root_pos_w[:, :3]
    # print(pos,"pos")
    # print(pos_box,"box")

    # ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return torch.zeros((env.num_envs),device= env.device)



def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame.

    Uses exponential kernel for reward computation.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_robot_close_to_box_before_x_minus05_exp(
    env: ManagerBasedRLEnv,
    std: float=1.5,
    box_asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Stage1 reward: before box x > -0.5, reward robot for getting close to box in the XY plane.
    Once box x > -0.5, reward becomes zero.
    The closer robot XY to box center, the higher the exponential reward.
    """
    box: RigidObject = env.scene[box_asset_cfg.name]
    robot: RigidObject = env.scene[robot_asset_cfg.name]

    # World position [num_envs, 3]
    box_pos = box.data.root_pos_w
    robot_pos = robot.data.root_pos_w

    box_x, box_y = box_pos[:, 0], box_pos[:, 1]
    robot_x, robot_y = robot_pos[:, 0], robot_pos[:, 1]

    # XY distance between robot and box center
    dx = robot_x - box_x
    dy = robot_y - box_y

    # Euclidean distance on XY plane
    dist_xy_sq = dx ** 2 + dy ** 2

    # Exponential reward based directly on center distance
    reward_xy = torch.exp(-dist_xy_sq / (std ** 2))

    # Only activate reward when box x <= -0.5
    box_not_passed_mask = box_x <= -0.5
    final_reward = torch.where(
        box_not_passed_mask,
        reward_xy,
        torch.zeros_like(reward_xy)
    )

    return final_reward


def track_robot_head_to_box_before_x_minus05_exp(
    env: ManagerBasedRLEnv,
    target_forward: float = 0.9,
    std_forward: float = 0.6,
    std_lateral: float = 0.25,
    box_asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Stage1 reward: before box x > -0.5, reward the robot for approaching the box head-on.
    The reward is highest when the box is in front of the robot in the robot yaw frame,
    with a small lateral offset, discouraging side-on contact.
    """
    box: RigidObject = env.scene[box_asset_cfg.name]
    robot: RigidObject = env.scene[robot_asset_cfg.name]

    box_pos = box.data.root_pos_w
    robot_pos = robot.data.root_pos_w

    box_x = box_pos[:, 0]

    rel_pos_w = box_pos - robot_pos
    rel_pos_body = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)

    rel_x = rel_pos_body[:, 0]
    rel_y = rel_pos_body[:, 1]

    forward_reward = torch.exp(-torch.square(rel_x - target_forward) / (std_forward ** 2))
    lateral_reward = torch.exp(-torch.square(rel_y) / (std_lateral ** 2))

    # Only reward when the box is in front of the robot.
    in_front_mask = rel_x > 0.0
    reward_head_on = forward_reward * lateral_reward * in_front_mask.to(forward_reward.dtype)

    box_not_passed_mask = box_x <= -0.5
    final_reward = torch.where(
        box_not_passed_mask,
        reward_head_on,
        torch.zeros_like(reward_head_on),
    )

    return final_reward


def track_robot_behind_box_toward_target_exp(
    env: ManagerBasedRLEnv,
    target_x: float = 0.0,
    target_y: float = -1.0,
    target_distance: float = 0.85,
    std_longitudinal: float = 0.45,
    std_lateral: float = 0.25,
    activate_box_x_threshold: float = -0.5,
    box_asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Before the box reaches the pit entry, reward the robot for moving to the
    back side of the box relative to the box->target direction.
    """
    box: RigidObject = env.scene[box_asset_cfg.name]
    robot: RigidObject = env.scene[robot_asset_cfg.name]

    box_pos = box.data.root_pos_w[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    target_xy = _target_xy_like(box_pos, target_x=target_x, target_y=target_y)

    push_dir = _safe_normalize_xy(target_xy - box_pos)
    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)
    desired_robot_xy = box_pos - target_distance * push_dir
    pose_error_xy = robot_pos - desired_robot_xy

    longitudinal_error = torch.sum(pose_error_xy * push_dir, dim=1)
    lateral_error = torch.sum(pose_error_xy * lateral_dir, dim=1)

    reward = torch.exp(-torch.square(longitudinal_error) / (std_longitudinal**2))
    reward *= torch.exp(-torch.square(lateral_error) / (std_lateral**2))

    box_x = box.data.root_pos_w[:, 0]
    active_mask = box_x <= activate_box_x_threshold
    return torch.where(active_mask, reward, torch.zeros_like(reward))


def track_robot_heading_to_push_target_exp(
    env: ManagerBasedRLEnv,
    target_x: float = 0.0,
    target_y: float = -1.0,
    std_forward: float = 0.3,
    std_lateral: float = 0.25,
    close_dist: float = 1.4,
    activate_box_x_threshold: float = -0.5,
    box_asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward the robot for facing the box->target push direction once it is close
    enough to the box to start a meaningful push.
    """
    box: RigidObject = env.scene[box_asset_cfg.name]
    robot: RigidObject = env.scene[robot_asset_cfg.name]

    box_pos = box.data.root_pos_w[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    target_xy = _target_xy_like(box_pos, target_x=target_x, target_y=target_y)

    push_dir_xy = _safe_normalize_xy(target_xy - box_pos)
    push_dir_w = torch.cat((push_dir_xy, torch.zeros_like(push_dir_xy[:, :1])), dim=1)
    push_dir_body = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), push_dir_w)

    forward_reward = torch.exp(-torch.square(push_dir_body[:, 0] - 1.0) / (std_forward**2))
    lateral_reward = torch.exp(-torch.square(push_dir_body[:, 1]) / (std_lateral**2))

    robot_box_dist = torch.linalg.norm(robot_pos - box_pos, dim=1)
    close_mask = robot_box_dist <= close_dist
    box_x = box.data.root_pos_w[:, 0]
    active_mask = torch.logical_and(box_x <= activate_box_x_threshold, close_mask)

    reward = forward_reward * lateral_reward
    return torch.where(active_mask, reward, torch.zeros_like(reward))


def track_box_velocity_toward_target_exp(
    env: ManagerBasedRLEnv,
    target_x: float = 0.0,
    target_y: float = -1.0,
    speed_scale: float = 0.2,
    std_lateral: float = 0.25,
    robot_box_contact_dist: float = 1.3,
    activate_box_x_threshold: float = -0.5,
    box_asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward box motion toward the pit target and suppress sideways sliding.
    """
    box: RigidObject = env.scene[box_asset_cfg.name]
    robot: RigidObject = env.scene[robot_asset_cfg.name]

    box_pos = box.data.root_pos_w[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    target_xy = _target_xy_like(box_pos, target_x=target_x, target_y=target_y)

    push_dir = _safe_normalize_xy(target_xy - box_pos)
    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)
    box_vel_xy = box.data.root_lin_vel_w[:, :2]

    forward_speed = torch.sum(box_vel_xy * push_dir, dim=1)
    lateral_speed = torch.sum(box_vel_xy * lateral_dir, dim=1)

    forward_reward = 1.0 - torch.exp(-torch.clamp(forward_speed, min=0.0) / speed_scale)
    lateral_reward = torch.exp(-torch.square(lateral_speed) / (std_lateral**2))

    robot_box_dist = torch.linalg.norm(robot_pos - box_pos, dim=1)
    contact_scale = torch.exp(-torch.square(robot_box_dist - 0.85) / (0.55**2))

    box_x = box.data.root_pos_w[:, 0]
    active_mask = torch.logical_and(box_x <= activate_box_x_threshold, robot_box_dist <= robot_box_contact_dist)
    reward = forward_reward * lateral_reward * contact_scale
    return torch.where(active_mask, reward, torch.zeros_like(reward))


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(
#         env.action_manager.action - 2 * env.action_manager.prev_action
#         + env.action_manager.prev_prev_action
#     )
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
