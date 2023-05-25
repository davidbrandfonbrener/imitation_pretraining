"""Sweep for bc training on all tasks."""
import os
import fire

from data_configs import finteune_data_configs
from sweep_utils import (
    write_slurm_file,
    grid_to_list,
    log_gridlist_and_config,
    update_data_dirs,
)

from imitation_pretraining.experiments.training import train
from configs import bc

MODE = "GPU"
HRS = 4
ABLATE_DATA_SIZE = True
SEEDS = 5

SWEEP_GRID = dict(
    pretrain=[False],
    num_steps=[10000],
    rollout_freq=[10000],
    num_rollouts=[100],
    policy_network_name=["policy-conv-softmax-64-norm-0.1"],
    observation_adapter_name=["pixels_crop"],
)
SWEEP_LIST = grid_to_list(SWEEP_GRID)

DATA_LIST = finteune_data_configs(ABLATE_DATA_SIZE)

FULL_SWEEP_LIST = []
for data_config in DATA_LIST:
    for sweep_config in SWEEP_LIST:
        full_config = dict(data_config, **sweep_config)
        if "kitchen" in full_config["eval_env_name"]:
            full_config["num_steps"] = 30000
            full_config["rollout_freq"] = 30000
            full_config["num_rollouts"] = 50
        FULL_SWEEP_LIST.append(full_config)

print("Total jobs:", len(FULL_SWEEP_LIST))


def main(idx: int, sweep_id: int = 0, test: bool = False):
    """Launch sweep job."""
    if idx == 0:
        write_slurm_file(
            len(FULL_SWEEP_LIST), os.path.basename(__file__), mode=MODE, hrs=HRS
        )
    else:
        for seed in range(SEEDS):
            config = bc.get_config()  # Get base config
            config.update(FULL_SWEEP_LIST[idx - 1])  # Update based on index
            config["seed"] = int(1e6) + seed
            config = update_data_dirs(config)

            id = idx + seed * len(FULL_SWEEP_LIST)
            config["job_id"] = id
            config["sweep_id"] = sweep_id
            if id == 1 and not test:  # Log sweep grid
                log_gridlist_and_config(FULL_SWEEP_LIST, config)
            train.run(config, test=test)


if __name__ == "__main__":
    fire.Fire(main)
