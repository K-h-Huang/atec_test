import os
from typing import Any

import torch


class AlgSolution:
    """Deploy the TaskD high-level velocity policy with the B2 low-level leg policy."""

    LEG_ACTION_DIM = 12
    ARM_ACTION_DIM = 8

    LEG_JOINT_INDICES = list(range(12))
    ARM_JOINT_INDICES = list(range(12, 20))

    def __init__(self):
        solution_dir = os.path.dirname(os.path.abspath(__file__))

        high_level_policy_path = os.path.join(solution_dir, "taske", "policy.pt")
        low_level_policy_path = os.path.join(solution_dir, "policy.pt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.high_level_policy = torch.jit.load(high_level_policy_path, map_location=self.device)
        self.high_level_policy.eval()

        self.low_level_policy = torch.jit.load(low_level_policy_path, map_location=self.device)
        self.low_level_policy.eval()

        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.last_velocity_commands = None

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def _split_proprio(
        self, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split official B2Piper proprio into the leg parts used by the training configs."""
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        if action_dim < self.LEG_ACTION_DIM:
            raise ValueError(f"Expected at least {self.LEG_ACTION_DIM} action dims, got {action_dim}.")

        idx = 0

        has_base_lin_vel = proprio.shape[-1] == 12 + action_dim * 3
        if has_base_lin_vel:
            idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.LEG_JOINT_INDICES]
        joint_vel_leg = joint_vel_all[:, self.LEG_JOINT_INDICES]
        actions_env_leg = actions_all[:, self.LEG_JOINT_INDICES]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        return base_ang_vel, projected_gravity, joint_pos_leg, joint_vel_leg, actions_train_leg

    def _get_last_velocity_commands(self, num_envs: int, dtype: torch.dtype) -> torch.Tensor:
        if self.last_velocity_commands is None or self.last_velocity_commands.shape[0] != num_envs:
            self.last_velocity_commands = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        return self.last_velocity_commands.to(device=self.device, dtype=dtype)

    def _build_high_level_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        extero = obs.get("extero")

        if extero is None:
            raise KeyError("TaskD high-level policy was trained with extero lidar observations, but obs['extero'] is missing.")

        base_ang_vel, projected_gravity, joint_pos_leg, joint_vel_leg, _actions_train_leg = self._split_proprio(proprio)
        last_velocity_commands = self._get_last_velocity_commands(proprio.shape[0], proprio.dtype)
        height_scan = torch.clamp(extero.to(self.device), min=-1.0, max=1.0)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                last_velocity_commands,
                height_scan,
            ],
            dim=-1,
        )

    def _build_low_level_obs(self, obs: dict[str, torch.Tensor], velocity_commands: torch.Tensor) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        base_ang_vel, projected_gravity, joint_pos_leg, joint_vel_leg, actions_train_leg = self._split_proprio(proprio)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

    def _map_low_level_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.LEG_ACTION_DIM:
            raise ValueError(
                f"Low-level policy output dim mismatch: got {action_train.shape[-1]}, expected {self.LEG_ACTION_DIM}."
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        action_env[:, self.LEG_JOINT_INDICES] = leg_action_env

        if action_dim >= self.LEG_ACTION_DIM + self.ARM_ACTION_DIM:
            action_env[:, self.ARM_JOINT_INDICES] = 0.0

        return action_env

    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        with torch.inference_mode():
            high_level_obs = self._build_high_level_obs(obs)
            velocity_commands = self.high_level_policy(high_level_obs).to(device=self.device, dtype=torch.float32)
            if velocity_commands.ndim == 1:
                velocity_commands = velocity_commands.unsqueeze(0)
            velocity_commands = torch.clamp(velocity_commands, min=-3.0, max=3.0)

            low_level_obs = self._build_low_level_obs(obs, velocity_commands)
            low_level_action = self.low_level_policy(low_level_obs)
            if low_level_action.ndim == 1:
                low_level_action = low_level_action.unsqueeze(0)

            action_env = self._map_low_level_action_to_env_action(low_level_action.to(dtype=torch.float32), action_dim)

        self.last_velocity_commands = velocity_commands.detach()

        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}
