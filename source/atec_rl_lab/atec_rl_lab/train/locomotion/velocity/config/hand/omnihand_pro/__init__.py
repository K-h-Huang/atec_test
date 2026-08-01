"""OmniHand Pro velocity locomotion tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="ATEC-Isaac-Velocity-Rough-OmniHandPro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:OmniHandProRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniHandProRoughPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-OmniHandPro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:OmniHandProFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniHandProFlatPPORunnerCfg",
    },
)

from .flat_env_cfg import OmniHandProFlatEnvCfg
from .rough_env_cfg import OmniHandProRoughEnvCfg

__all__ = ["OmniHandProFlatEnvCfg", "OmniHandProRoughEnvCfg"]
