"""Sweep for bc training on all tasks."""
import os
import fire

from data_configs import finteune_data_configs
from sweep_utils import (
    write_slurm_file,
    grid_to_list,
    log_gridlist_and_config,
    update_data_dirs,
    find_encoder_matches,
    ROOT_DIR,
)

from imitation_pretraining.experiments.training import train
from configs import bc
from imitation_pretraining.experiments.evaluation import loading

MODE = "CPU"
HRS = 3

PRETRAIN_DATE = "Apr-2023"
CKPT_DIR = os.path.join(ROOT_DIR, "ckpts", PRETRAIN_DATE)

PRETRAIN_SWEEP_ID = "32658444"
ENCODER_CONFIGS = loading.load_sweep_configs(ROOT_DIR, PRETRAIN_DATE, PRETRAIN_SWEEP_ID)
ENCODER_CONFIGS = [e for e in ENCODER_CONFIGS if e["job_id"] <= 45]

PRETRAIN_SWEEP_ID = "32745609"
ENCODER_CONFIGS += loading.load_sweep_configs(
    ROOT_DIR, PRETRAIN_DATE, PRETRAIN_SWEEP_ID
)

print(f"Number of configs: {len(ENCODER_CONFIGS)}")


SWEEP_GRID = dict(
    pretrain=[False],
    seed=[int(1e6) + i for i in range(5)],
    observation_adapter_name=["embedding"],
    policy_network_name=["policy-mlp-256-0.1"],
    encode_data=[True],
)
SWEEP_LIST = grid_to_list(SWEEP_GRID)

DATA_LIST = finteune_data_configs(sweep_size=False)

FULL_SWEEP_LIST = []
for data_config in DATA_LIST:
    for sweep_config in SWEEP_LIST:

        full_config = dict(data_config, **sweep_config)
        if "kitchen" in full_config["eval_env_name"]:
            full_config["num_rollouts"] = 50

        encoder_matches = find_encoder_matches(full_config, ENCODER_CONFIGS)

        # Ensure we have 5 algs x 3 data sizes = 15 matches
        n_match = len(encoder_matches)
        assert n_match == 15, f"Found {n_match} matches for {full_config}"

        for encoder_config in encoder_matches:
            ckpt_dir = os.path.join(
                CKPT_DIR, str(encoder_config["sweep_id"]), str(encoder_config["job_id"])
            )
            if not os.path.exists(ckpt_dir + "/state"):
                print(f"Missing {ckpt_dir}")
            encoder_config["checkpoint_path"] = ckpt_dir

            matched_config = dict(
                full_config,
                encoder_name=encoder_config["agent_name"],
                encoder_config=encoder_config,
            )

            FULL_SWEEP_LIST.append(matched_config)

print("Total jobs:", len(FULL_SWEEP_LIST))


def main(idx: int, sweep_id: int = 0, test: bool = False):
    """Launch sweep job."""
    if idx == 0:
        write_slurm_file(
            len(FULL_SWEEP_LIST), os.path.basename(__file__), mode=MODE, hrs=HRS
        )
    else:
        config = bc.get_config()  # Get base config
        config.update(FULL_SWEEP_LIST[idx - 1])  # Update based on index
        config = update_data_dirs(config)
        config["job_id"] = idx
        config["sweep_id"] = sweep_id
        if idx == 1 and not test:  # Log sweep grid
            log_gridlist_and_config(FULL_SWEEP_LIST, config)
        train.run(config, test=test)


if __name__ == "__main__":
    fire.Fire(main)
