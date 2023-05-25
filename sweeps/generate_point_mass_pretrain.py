"""A test sweep for bc pretraining."""
import os
import fire

from sweep_utils import write_slurm_file, grid_to_list

from configs import generate_point_mass
from imitation_pretraining.experiments.data_generation import point_mass

GRID_LIST = []

# Pretrain standard
GRID = {
    "seed": [0],
    "episodes": [1000],
    "episodes_per_seed": [1],
    "env_name": ["point_mass"],
    "policy_type": ["expert"],
    "pretrain": [True],
}
assert set(GRID.keys()).issubset(list(generate_point_mass.get_config()))
GRID_LIST.extend(grid_to_list(GRID))


def main(idx, sweep_id=0):
    del sweep_id
    if idx == 0:
        write_slurm_file(len(GRID_LIST), os.path.basename(__file__), mode="GPU", hrs=2)
    else:
        config = generate_point_mass.get_config()
        config.update(GRID_LIST[idx - 1])
        config["device"] = "cpu"
        point_mass.PointMassGenerator(config).generate()


if __name__ == "__main__":
    fire.Fire(main)
