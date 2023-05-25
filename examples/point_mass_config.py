"""Data generation config for point mass."""
import os
from pathlib import Path


def get_config():
    """Returns the default config for data generation."""
    root_path = Path(os.path.dirname(os.path.realpath(__file__)))
    config = {
        "episodes": 10,
        "episodes_per_seed": 1,
        "seed": 0,
        "env_name": "point_mass",
        "root_path": root_path,
        "clean_dir": True,
        "policy_type": "expert",
        "pretrain": True,
        "success_only": False,
    }
    return config
