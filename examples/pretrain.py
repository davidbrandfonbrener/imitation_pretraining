"""Example of pretraining using BC."""
import fire
import id_config
from imitation_pretraining.experiments.training import train


def main():
    config = id_config.get_config()

    # Set data directories (as in sweep_utils.py)
    env_name = config["eval_env_name"]
    pretrain_name = "pretrain"
    config["ep"] = 10
    config["per"] = 1
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
    config["sweep_id"] = 0
    config["job_id"] = 0

    train.run(config, test=True)  # test=True diables wandb logging


if __name__ == "__main__":
    fire.Fire(main)
