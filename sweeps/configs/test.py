"""Smaller test config."""
import configs


def get_config(agent_name, local=False):
    """Returns the default config for CD pretraining."""
    config = configs.get_config(agent_name, local=local)

    # Shorten training time
    config["num_steps"] = 10
    config["batch_size"] = 32
    config["eval_freq"] = 5
    config["rollout_freq"] = 5
    config["num_rollouts"] = 1
    env_name = "point_mass"
    config["eval_env_name"] = env_name
    # Data paths
    config["train_data_dir"] = (
        config["project_dir"]
        / "data"
        / f"{env_name}/pretrain/ep-100-per-1-seed-0-expert"
    )
    config["eval_data_dir"] = (
        config["project_dir"]
        / "data"
        / f"{env_name}/pretrain/ep-100-per-1-seed-0-expert-eval"
    )
    return config
