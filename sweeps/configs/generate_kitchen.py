"""Data generation config for d4rl."""
import config_utils


def get_config(local=False):
    """Returns the default config for data generation."""
    root_path = config_utils.get_root_path(local)
    config = {
        "env_name": "kitchen_split_0",
        "root_path": root_path,
        "clean_dir": True,
        "pretrain": False,
        "train_eps": 15,
        "eval_eps": 50,
        "seed": 0,
    }
    return config
