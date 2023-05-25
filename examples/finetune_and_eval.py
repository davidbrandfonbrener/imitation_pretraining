"""Example of finetuning using BC."""
import os
import time
import fire
import bc_config
from imitation_pretraining.experiments.training import train
from imitation_pretraining.experiments.evaluation import loading


def main():
    config = bc_config.get_config()

    # Load the pretrained encoder
    root_dir = config["project_dir"]
    date = time.strftime("%b-%Y")
    encoder_sweep_id = "0"
    encoder_config = loading.load_sweep_configs(root_dir, date, encoder_sweep_id)[0]
    ckpt_dir = os.path.join(
        root_dir,
        "ckpts",
        date,
        str(encoder_config["sweep_id"]),
        str(encoder_config["job_id"]),
    )
    if not os.path.exists(ckpt_dir + "/state"):
        print(f"Missing {ckpt_dir}")
    encoder_config["checkpoint_path"] = ckpt_dir

    config["encoder_name"] = encoder_config["agent_name"]
    config["encoder_config"] = encoder_config

    # Set data directories (as in sweep_utils.py)
    env_name = config["eval_env_name"]
    pretrain_name = "finetune"
    config["ep"] = 10
    config["per"] = 10
    config["train_data_dir"] = (
        config["project_dir"]
        / f"data/{env_name}/{pretrain_name}/"
        / f"ep-{config['ep']}-per-{config['per']}-seed-{config['seed']}-expert"
    )
    config["eval_data_dir"] = (
        config["project_dir"]
        / f"data/{env_name}/{pretrain_name}/"
        / f"ep-{config['ep']}-per-{config['per']}-seed-{config['seed']}-expert-eval"
    )

    # Set IDs which determine path for logs and checkpoints
    config["sweep_id"] = 1
    config["job_id"] = 0

    train.run(config, test=True)  # test=True diables wandb logging


if __name__ == "__main__":
    fire.Fire(main)
