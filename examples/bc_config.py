"""Default config for BC."""
import os
from pathlib import Path


def get_config():
    """Returns the default config for BC."""
    root_path = Path(os.path.dirname(os.path.realpath(__file__)))
    config = {
        # Names
        "agent_name": "bc",
        # Paths
        "project_dir": root_path,
        "train_data_dir": root_path / "???",
        "eval_data_dir": root_path / "???",
        "checkpoint_path": None,
        # Training hyperparams
        "seed": 0,
        "observation_adapter_name": "embedding",
        "policy_network_name": "policy-mlp-256-0.1",
        "target_predictor_network_name": "target-mlp",
        "lr": 1e-3,
        "predict_target": False,
        "num_steps": 1000,
        "batch_size": 256,
        "max_episodes": 10000,
        "log_freq": 10,
        "n_tasks": None,
        "nstep": 1,
        "history": 1,
        "n_bins": 1,
        # Encoder params
        "encoder_name": None,
        "encode_data": True,
        "encoder_config": {},
        "finetune_from_encoder": False,
        # Eval hyperparams
        "eval_freq": 100,
        "eval_env_name": "point_mass",
        "pretrain": False,
        "rollout_freq": 1000,
        "num_rollouts": 10,
        "fix_eval_task": False,
        # Sweep parameters
        "job_id": 0,
        "sweep_id": 0,
    }
    return config
