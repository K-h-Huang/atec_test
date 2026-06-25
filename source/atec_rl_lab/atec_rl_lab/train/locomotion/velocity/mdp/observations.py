# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def obs_box_rel_robot(env: ManagerBasedRLEnv,
                      box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
                      robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Observation term: box position relative to robot world frame (raw, no normalization).
    Output shape: [num_envs, 3] = (box_x - robot_x, box_y - robot_y, box_z - robot_z)
    """
    box: RigidObject = env.scene[box_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    box_w = box.data.root_pos_w    # [N,3] box world center
    robot_w = robot.data.root_pos_w# [N,3] robot world center

    rel_pos = box_w - robot_w
    return rel_pos


# def visualizable_image(
#     env: ManagerBasedEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera") )-> torch.Tensor:
    
#     sensor: TiledCamera | Camera | RayCasterCamera = (
#         env.scene.sensors[sensor_cfg.name]
#     )
#     data = sensor.data