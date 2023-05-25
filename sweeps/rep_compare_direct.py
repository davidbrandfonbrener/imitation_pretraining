"""Sweep for pretraining."""
import os
import fire

from pretrain_alg_configs import alg_configs
from sweep_utils import (
    write_slurm_file,
    log_gridlist_and_config,
    update_data_dirs,
    ROOT_DIR,
)

from imitation_pretraining.experiments.training import train_double_encoder as train
from imitation_pretraining.experiments.evaluation import loading
from configs import bc

MODE = "CPU"
HRS = 1

ALG_LIST = alg_configs(identity_encoder=True)
COMMON_DICT = dict(
    observation_adapter_name="embedding",
    encode_data=True,
    pretrain=True,
    policy_network_name="policy-mlp-256-0.1",
    num_steps=10000,
    rollout_freq=1000000,  # Do not rollout
    num_rollouts=0,  # Do not rollout
)

PRETRAIN_DATE = "Apr-2023"
CKPT_DIR = os.path.join(ROOT_DIR, "ckpts", PRETRAIN_DATE)

PRETRAIN_SWEEP_ID = "32658444"
ENCODER_CONFIGS = loading.load_sweep_configs(ROOT_DIR, PRETRAIN_DATE, PRETRAIN_SWEEP_ID)
ENCODER_CONFIGS = [e for e in ENCODER_CONFIGS if e["job_id"] <= 45]

PRETRAIN_SWEEP_ID = "32745609"
ENCODER_CONFIGS += loading.load_sweep_configs(
    ROOT_DIR, PRETRAIN_DATE, PRETRAIN_SWEEP_ID
)

SIZE_FILTER = {
    "point_mass": 1000,
    "metaworld_pick_place_nogoal": 1000,
    "kitchen_split_0": 450,
    "metaworld_pretrain_split_door": 1000,
    "metaworld_pretrain_split_0": 10000,
    "metaworld_pretrain_split_r3m": 10000,
}
ENCODER_CONFIGS = [
    e for e in ENCODER_CONFIGS if e["max_episodes"] == SIZE_FILTER[e["eval_env_name"]]
]
print(f"Number of encoders: {len(ENCODER_CONFIGS)}")

FULL_SWEEP_LIST = []
for encoder_config in ENCODER_CONFIGS:
    for target_config in ENCODER_CONFIGS:
        # Ensure encoders use same dataset
        if encoder_config["eval_env_name"] != target_config["eval_env_name"]:
            continue
        if encoder_config["max_episodes"] != target_config["max_episodes"]:
            continue

        full_config = COMMON_DICT.copy()
        full_config["encoder_name"] = encoder_config["agent_name"]
        encoder_config["checkpoint_path"] = os.path.join(
            CKPT_DIR, str(encoder_config["sweep_id"]), str(encoder_config["job_id"])
        )
        full_config["encoder_config"] = encoder_config

        full_config["target_encoder_name"] = target_config["agent_name"]
        target_config["checkpoint_path"] = os.path.join(
            CKPT_DIR, str(target_config["sweep_id"]), str(target_config["job_id"])
        )
        full_config["target_encoder_config"] = target_config

        # Use same dataset as encoder to finetune
        full_config["ep"] = encoder_config["ep"]
        full_config["per"] = encoder_config["per"]
        full_config["seed"] = encoder_config["seed"]
        full_config["eval_env_name"] = encoder_config["eval_env_name"]

        full_config["max_episodes"] = 200  # Smaller dataset for finetuning

        FULL_SWEEP_LIST.append(full_config)

print("Total jobs (without state jobs):", len(FULL_SWEEP_LIST))

for encoder_config in ENCODER_CONFIGS:
    adapter_name = "position_object"
    if "point_mass" in encoder_config["eval_env_name"]:
        adapter_name = "position"
    if "metaworld_pretrain" in encoder_config["eval_env_name"]:
        adapter_name = "position_goal_object"

    # State as source (encoder=None, obs_adapter from state)
    full_config = dict(COMMON_DICT, observation_adapter_name=adapter_name)
    full_config["encoder_name"] = None
    full_config["encoder_config"] = None

    full_config["target_encoder_name"] = encoder_config["agent_name"]
    encoder_config["checkpoint_path"] = os.path.join(
        CKPT_DIR, str(encoder_config["sweep_id"]), str(encoder_config["job_id"])
    )
    full_config["target_encoder_config"] = encoder_config

    # Use same dataset as encoder to finetune
    full_config["ep"] = encoder_config["ep"]
    full_config["per"] = encoder_config["per"]
    full_config["seed"] = encoder_config["seed"]
    full_config["eval_env_name"] = encoder_config["eval_env_name"]
    full_config["max_episodes"] = 200  # Smaller dataset for finetuning
    FULL_SWEEP_LIST.append(full_config)

    # State as target (encoder=config, target_encoder=func that returns obs adapter)
    full_config = COMMON_DICT.copy()
    full_config["encoder_name"] = encoder_config["agent_name"]
    encoder_config["checkpoint_path"] = os.path.join(
        CKPT_DIR, str(encoder_config["sweep_id"]), str(encoder_config["job_id"])
    )
    full_config["encoder_config"] = encoder_config
    full_config["target_encoder_name"] = None
    full_config["target_encoder_config"] = {"observation_adapter_name": adapter_name}

    # Use same dataset as encoder to finetune
    full_config["ep"] = encoder_config["ep"]
    full_config["per"] = encoder_config["per"]
    full_config["seed"] = encoder_config["seed"]
    full_config["eval_env_name"] = encoder_config["eval_env_name"]
    full_config["max_episodes"] = 200  # Smaller dataset for finetuning
    FULL_SWEEP_LIST.append(full_config)

    # Add state as source and target (encoder=None, target_encoder=func that returns obs adapter)
    if "reconstruction" == encoder_config["agent_name"]:
        full_config = dict(COMMON_DICT, observation_adapter_name=adapter_name)
        full_config["encoder_name"] = None
        full_config["encoder_config"] = None
        full_config["target_encoder_name"] = None
        full_config["target_encoder_config"] = {
            "observation_adapter_name": adapter_name
        }

        # Use same dataset as encoder to finetune
        full_config["ep"] = encoder_config["ep"]
        full_config["per"] = encoder_config["per"]
        full_config["seed"] = encoder_config["seed"]
        full_config["eval_env_name"] = encoder_config["eval_env_name"]
        full_config["max_episodes"] = 200
        FULL_SWEEP_LIST.append(full_config)
        print(len(FULL_SWEEP_LIST))


print("Total jobs (with state jobs):", len(FULL_SWEEP_LIST))


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
        print(config)
        train.run(config, test=test)


if __name__ == "__main__":
    fire.Fire(main)
