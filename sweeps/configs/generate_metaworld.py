"""Data generation config for metaworld."""
import config_utils


def get_config(local=False):
    """Returns the default config for data generation."""
    root_path = config_utils.get_root_path(local)
    config = {
        "episodes": 10,
        "episodes_per_seed": 1,
        "seed": 0,
        "env_name": "metaworld_pretrain_split_1",
        "root_path": root_path,
        "clean_dir": True,
        "policy_type": "expert",
        "pretrain": True,
        "success_only": False,
    }
    return config
