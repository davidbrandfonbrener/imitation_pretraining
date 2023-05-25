"""A test sweep for bc pretraining."""
import os
import fire

from sweep_utils import write_slurm_file, grid_to_list

from configs import generate_metaworld
from imitation_pretraining.experiments.data_generation import metaworld


GRID_LIST = []

GRID = {
    "seed": [int(1e6 + i) for i in range(5)],
    "episodes": [20],
    "episodes_per_seed": [20],
    "env_name": ["metaworld_pick_place_nogoal"],
    "pretrain": [False],
    "policy_type": ["expert"],
}
assert set(GRID.keys()).issubset(list(generate_metaworld.get_config()))
GRID_LIST.extend(grid_to_list(GRID))

GRID = {
    "seed": [int(1e6 + i) for i in range(5)],
    "episodes": [20],
    "episodes_per_seed": [1],
    "env_name": [
        f"metaworld_finetune_{j}_split_{s}" for j in range(5) for s in [0, 1, 2, "r3m"]
    ]
    + [f"metaworld_finetune_0_split_{s}" for s in ["plate", "button", "door"]],
    "pretrain": [False],
}
assert set(GRID.keys()).issubset(list(generate_metaworld.get_config()))
GRID_LIST.extend(grid_to_list(GRID))

# BDP-all data
GRID = {
    "seed": [int(1e6 + i) for i in range(5)],
    "episodes": [20],
    "episodes_per_seed": [1],
    "env_name": [
        f"metaworld_finetune_0_split_{s}-all" for s in ["plate", "button", "door"]
    ],
    "pretrain": [False],
}
assert set(GRID.keys()).issubset(list(generate_metaworld.get_config()))
GRID_LIST.extend(grid_to_list(GRID))


print("Total jobs:", len(GRID_LIST))


def main(idx, sweep_id=0):
    del sweep_id
    if idx == 0:
        # Do not use gpu
        write_slurm_file(len(GRID_LIST), os.path.basename(__file__), mode="CPU", hrs=1)
    else:
        config = generate_metaworld.get_config()
        config.update(GRID_LIST[idx - 1])
        config["device"] = "cpu"
        metaworld.MetaWorldGenerator(config).generate()


if __name__ == "__main__":
    fire.Fire(main)
