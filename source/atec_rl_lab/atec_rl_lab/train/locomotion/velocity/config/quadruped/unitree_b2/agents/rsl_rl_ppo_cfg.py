# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoActorCriticCNNCfg(RslRlPpoActorCriticCfg):
    actor_cnn_cfg: dict | None = None
    critic_cnn_cfg: dict | None = None


@configclass
class UnitreeB2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2FlatPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_b2_flat"


@configclass
class UnitreeB2TaskBPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_b2_taskb"
    obs_groups = {"policy": ["proprio", "extero", "image"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCNNCfg(
        class_name="ActorCriticCNN",
        init_noise_std=0.6,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_cnn_cfg={
            "output_channels": [16, 32, 64, 64],
            "kernel_size": [8, 4, 3, 3],
            "stride": [4, 2, 2, 2],
            "padding": "zeros",
            "activation": "elu",
            "global_pool": "avg",
            "flatten": True,
        },
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
