"""Data generation config for point mass goal based pixels."""
import config_utils


def get_config(local=False):
    """Returns the default config for data generation."""
    root_path = config_utils.get_root_path(local)
    config = {
        "episodes": 10,
        "episodes_per_seed": 10,
        "seed": 0,
        "env_name": "point_mass",
        "root_path": root_path,
        "clean_dir": True,
        "policy_type": "expert",
        "pretrain": True,
        "success_only": False,
    }
    return config
