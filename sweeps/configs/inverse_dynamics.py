"""Default config for inverse dynamics pretraining."""
import config_utils


def get_config(local=False):
    """Returns the default config for pretraining."""
    root_path = config_utils.get_root_path(local)
    config = {  # Names
        "agent_name": "inverse_dynamics",
        # Paths
        "project_dir": root_path,
        "train_data_dir": root_path / "???",
        "eval_data_dir": root_path / "???",
        "checkpoint_path": None,
        # Training hyperparams
        "seed": 0,
        "observation_adapter_name": "pixels",
        "phi_network_name": "encoder-conv-softmax-64-norm",
        "policy_network_name": "policy-mlp-256-0.1",
        "target_predictor_network_name": "target-mlp",
        "lr": 1e-3,
        "stop_grad": False,
        "predict_all_actions": False,
        "num_steps": 10000,
        "batch_size": 256,
        "max_episodes": 10000,
        "log_freq": 10,
        "n_tasks": None,
        "nstep": 1,
        "history": 1,
        # Encoder params
        "encoder_name": None,
        "encode_data": False,
        "encoder_config": {},
        # Eval hyperparams
        "eval_freq": 100,
        "eval_env_name": "point_mass",
        "rollout_freq": 1000000,  # Do not rollout
        "num_rollouts": 0,
        "fix_eval_task": False,
        # Sweep parameters
        "job_id": 0,
        "sweep_id": 0,
    }
    return config
