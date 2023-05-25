"""A test sweep for bc pretraining."""
import os
import fire

from sweep_utils import write_slurm_file, grid_to_list

from configs import generate_kitchen
from imitation_pretraining.experiments.data_generation import kitchen

WORKERS = 12
GRID_LIST = []

# Finetune
GRID = {
    "env_name": [f"kitchen_split_{i}" for i in range(3)],
    "pretrain": [False],
    "train_eps": [15],
    "seed": [int(1e6 + i) for i in range(5)],
}
GRID_LIST.extend(grid_to_list(GRID))

# Pretrain
GRID = {
    "env_name": ["kitchen_split_0"],
    "pretrain": [True],
    "train_eps": [450],
    "use_all_data": [True, False],
    "seed": [0],
}
GRID_LIST.extend(grid_to_list(GRID))

print("Total jobs:", len(GRID_LIST))


def main(idx, sweep_id=0):
    del sweep_id
    if idx == 0:
        write_slurm_file(
            len(GRID_LIST) * WORKERS, os.path.basename(__file__), mode="CPU", hrs=2
        )
    else:
        config = generate_kitchen.get_config()
        config.update(GRID_LIST[(idx - 1) // WORKERS])
        if WORKERS > 1:
            config["worker_id"] = idx - 1
            config["n_workers"] = WORKERS
        config["device"] = "cpu"
        kitchen.KitchenGenerator(config).generate()


if __name__ == "__main__":
    fire.Fire(main)
