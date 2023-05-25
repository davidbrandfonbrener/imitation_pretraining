"""Sweep for pretraining."""
import os
import fire

from data_configs import pretrain_data_all_tasks_configs
from pretrain_alg_configs import alg_configs
from sweep_utils import write_slurm_file, log_gridlist_and_config, update_data_dirs

from imitation_pretraining.experiments.training import train
import configs

MODE = "GPU"
HRS = 8

ALG_LIST = alg_configs()
DATA_LIST = pretrain_data_all_tasks_configs()
COMMON_DICT = dict(
    observation_adapter_name="pixels_crop",
    rollout_freq=1000000,  # Do not rollout
    num_rollouts=0,  # Do not rollout
)
SHORT_TRAIN_LIST = ["point_mass", "metaworld_pick_place_nogoal"] + [
    f"metaworld_pretrain_split_{s}" for s in ["button", "door", "plate"]
]

FULL_SWEEP_LIST = []
for data_config in DATA_LIST:
    for alg_config in ALG_LIST:
        full_config = dict(data_config, **alg_config, **COMMON_DICT)

        if full_config["eval_env_name"] in SHORT_TRAIN_LIST:
            full_config["num_steps"] = 100000
        else:
            full_config["num_steps"] = 200000

        if full_config["agent_name"] == "reconstruction":  # Need special params
            full_config["observation_adapter_name"] = "pixels"  # no augs
            full_config["batch_size"] = 128  # smaller batch
            if full_config["eval_env_name"] == "point_mass":
                full_config["decoder_network_name"] = "decoder-conv-84"
            else:
                full_config["decoder_network_name"] = "decoder-conv-120"

        FULL_SWEEP_LIST.append(full_config)


print("Total jobs:", len(FULL_SWEEP_LIST))


def main(idx: int, sweep_id: int = 0, test: bool = False):
    """Launch sweep job."""
    if idx == 0:
        write_slurm_file(
            len(FULL_SWEEP_LIST), os.path.basename(__file__), mode=MODE, hrs=HRS
        )
    else:
        alg = FULL_SWEEP_LIST[idx - 1]["agent_name"]
        config = configs.get_config(alg)
        config.update(FULL_SWEEP_LIST[idx - 1])  # Update based on index
        config = update_data_dirs(config)
        config["job_id"] = idx
        config["sweep_id"] = sweep_id
        if idx == 1 and not test:  # Log sweep grid
            log_gridlist_and_config(FULL_SWEEP_LIST, config)
        train.run(config, test=test)


if __name__ == "__main__":
    fire.Fire(main)
