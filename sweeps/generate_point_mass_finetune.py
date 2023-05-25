"""A test sweep for bc pretraining."""
import os
import fire

from sweep_utils import write_slurm_file, grid_to_list

from configs import generate_point_mass
from imitation_pretraining.experiments.data_generation import point_mass

GRID_LIST = []

# Finetune
GRID = {
    "seed": [int(1e6 + i) for i in range(10)],
    "episodes": [10],
    "episodes_per_seed": [10],
    "env_name": ["point_mass"],
    "policy_type": ["expert"],
    "pretrain": [False],
}
assert set(GRID.keys()).issubset(list(generate_point_mass.get_config()))
GRID_LIST.extend(grid_to_list(GRID))

print("Total jobs:", len(GRID_LIST))


def main(idx, sweep_id=0):
    del sweep_id
    if idx == 0:
        write_slurm_file(len(GRID_LIST), os.path.basename(__file__), mode="CPU", hrs=1)
    else:
        config = generate_point_mass.get_config()
        config.update(GRID_LIST[idx - 1])
        config["device"] = "cpu"
        point_mass.PointMassGenerator(config).generate()


if __name__ == "__main__":
    fire.Fire(main)
