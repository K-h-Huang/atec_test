"""Flat-ground velocity locomotion configuration for OmniHand Pro."""

from isaaclab.utils import configclass

from .rough_env_cfg import OmniHandProRoughEnvCfg


@configclass
class OmniHandProFlatEnvCfg(OmniHandProRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "OmniHandProFlatEnvCfg":
            self.disable_zero_weight_rewards()
