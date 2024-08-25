import os
from ray import tune
import copy

CONFIG = dict()

common_config = {
    "env": "MyEnv",
    "trainer_config": {},
    "env_config": {},
    "framework": "torch",
    "extra_python_environs_for_driver": {},
    "extra_python_environs_for_worker": {},
    "model": {
        "custom_model": "MyModel",
        "custom_model_config": {"value_function": None},
        "max_seq_len": 0,  # Placeholder value needed for ray to register model
    },
    "evaluation_config": {},
}

CONFIG["ppo"] = copy.deepcopy(common_config)
CONFIG["ppo"]["trainer_config"]["algorithm"] = "PPO"
CONFIG["ppo"].update(
    {
        "horizon": 10000,
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.99,
        "gamma": 0.99,
        "kl_coeff": 0.00,
        "shuffle_sequences": True,
        "num_sgd_iter": 3,
        "lr": 5e-5,
        "lr_schedule": None,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.000,
        "entropy_coeff_schedule": None,
        "clip_param": 0.2,
        "vf_clip_param": 100.0,
        "grad_clip": None,
        "kl_target": 0.01,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "normalize_actions": False,
        "clip_actions": True,
        # Device Configuration
        "create_env_on_driver": False,
        "num_cpus_for_driver": 0,
        "num_gpus": 1,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 2,
        "num_cpus_per_worker": 1,
    }
)
CONFIG["ppo"]["model"]["custom_model_config"]["sizes"] = {"policy": [512,512,512], "value": [512,512,512]}
CONFIG["ppo"]["model"]["custom_model_config"]["learningStd"] = False

CONFIG["ppo"]["sizes"] = CONFIG["ppo"]["model"]["custom_model_config"]["sizes"]
CONFIG["ppo"]["learningStd"] = CONFIG["ppo"]["model"]["custom_model_config"]["learningStd"]

# Muscle Configuration
CONFIG["ppo"]["trainer_config"]["muscle"] = {}
CONFIG["ppo"]["trainer_config"]["muscle"]["lr"] = 5e-5
CONFIG["ppo"]["trainer_config"]["muscle"]["num_epochs"] = 3
CONFIG["ppo"]["trainer_config"]["muscle"]["sizes"] = [256,256,256]
CONFIG["ppo"]["trainer_config"]["muscle"]["learningStd"] = True

# Large Set (For Cluster)
CONFIG["ppo_large"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_large"]["train_batch_size"] = 8192 * 8 * 4
CONFIG["ppo_large"]["sgd_minibatch_size"] = 4096
CONFIG["ppo_large"]["trainer_config"]["muscle"]["sgd_minibatch_size"] = 4096

# Medium Set (For a node or a PC)
CONFIG["ppo_medium"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_medium"]["train_batch_size"] = 8192 * 4
CONFIG["ppo_medium"]["sgd_minibatch_size"] = 4096
CONFIG["ppo_medium"]["trainer_config"]["muscle"]["sgd_minibatch_size"] = 4096

# Small Set
CONFIG["ppo_small"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_small"]["train_batch_size"] = 4096
CONFIG["ppo_small"]["sgd_minibatch_size"] = 512
CONFIG["ppo_small"]["trainer_config"]["muscle"]["sgd_minibatch_size"] = 512

# Mini Configuration (For DEBUG)
CONFIG["ppo_mini"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_mini"]["train_batch_size"] = 128
CONFIG["ppo_mini"]["sgd_minibatch_size"] = 64
CONFIG["ppo_mini"]["trainer_config"]["muscle"]["sgd_minibatch_size"] = 64

CONFIG["ppo_mini"]["num_workers"] = 1

# Large Set
CONFIG["ppo_large_server"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_server"]["num_workers"] = 128 * 2

CONFIG["ppo_large_node"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_node"]["num_workers"] = 64

CONFIG["ppo_large_pc"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_pc"]["num_workers"] = 32

# Medium Set
CONFIG["ppo_medium_server"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_server"]["num_workers"] = 128 * 2

CONFIG["ppo_medium_node"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_node"]["num_workers"] = 64

CONFIG["ppo_medium_pc"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_pc"]["num_workers"] = 32

# Small Set
CONFIG["ppo_small_server"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["ppo_small_server"]["num_workers"] = 128 * 2

CONFIG["ppo_small_node"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["ppo_small_node"]["num_workers"] = 64

CONFIG["ppo_small_pc"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["ppo_small_pc"]["num_workers"] = 16