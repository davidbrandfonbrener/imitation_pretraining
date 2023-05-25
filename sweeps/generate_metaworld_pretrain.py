"""A test sweep for bc pretraining."""
import os
import fire

from sweep_utils import write_slurm_file, grid_to_list

from configs import generate_metaworld
from imitation_pretraining.experiments.data_generation import metaworld

GRID_LIST = []

# Sinle task
GRID = {
    "seed": [0],
    "episodes": [1000],
    "episodes_per_seed": [1],
    "env_name": ["metaworld_pick_place_nogoal"],
    "pretrain": [True],
    "policy_type": ["expert"],
}
assert set(GRID.keys()).issubset(list(generate_metaworld.get_config()))
GRID_LIST.extend(grid_to_list(GRID))

# Small task families
GRID = {
    "seed": [0],
    "episodes": [1000],  # pretrain
    "episodes_per_seed": [1],
    "env_name": [f"metaworld_pretrain_split_{s}-all" for s in ["door"]]
    + [f"metaworld_pretrain_split_{s}" for s in ["door"]],
    "pretrain": [True],
}
assert set(GRID.keys()).issubset(list(generate_metaworld.get_config()))
GRID_LIST.extend(grid_to_list(GRID))

# Large task families
GRID = {
    "seed": [0],
    "episodes": [10000],  # pretrain
    "episodes_per_seed": [1],
    "env_name": [f"metaworld_pretrain_split_{i}" for i in ["r3m", 0, "all"]],
    "pretrain": [True],
}
assert set(GRID.keys()).issubset(list(generate_metaworld.get_config()))
GRID_LIST.extend(grid_to_list(GRID))

print("Total jobs:", len(GRID_LIST))


def main(idx, sweep_id=0):
    del sweep_id
    if idx == 0:
        write_slurm_file(len(GRID_LIST), os.path.basename(__file__), mode="CPU", hrs=6)
    else:
        config = generate_metaworld.get_config()
        config.update(GRID_LIST[idx - 1])
        config["device"] = "cpu"
        metaworld.MetaWorldGenerator(config).generate()


if __name__ == "__main__":
    fire.Fire(main)
