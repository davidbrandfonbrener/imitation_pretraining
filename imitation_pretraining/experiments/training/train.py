"""Basic training script for BC"""
import time

from imitation_pretraining import envs
from imitation_pretraining import algs
from imitation_pretraining.experiments.evaluation import eval as env_eval

from imitation_pretraining.experiments.training import training_utils
from imitation_pretraining.data_utils import replay_buffer


def _load_data(config, image_encoder=None):
    """Load data using params from config."""
    buffer = replay_buffer.ReplayBufferDataset(
        replay_dir=config["replay_dir"],
        max_episodes=config["max_episodes"],
        nstep=config["nstep"],
        discount=1.0,
        batch_size=config["batch_size"],
        seed=config["seed"],
        jax_device_queue=config.get("jax_device_queue", True),
        online=False,
        shuffle_and_repeat=True,
        cache=config.get("cache", False),
        history=config["history"],
        encoder=image_encoder,
        average_actions=config.get("average_actions", None),
    )
    iter_dataset = iter(buffer)
    init_batch = next(iter_dataset)
    return iter_dataset, init_batch


def run(config, test=False):
    """Pretrain the BC model using params from config."""
    # Load encoder if necessary to preprocess data
    if config["encoder_name"] is not None:
        _, init_batch = _load_data(dict(config, replay_dir=config["train_data_dir"]))
        fixed_encoder = algs.registry.make(
            config["encoder_name"], config=config["encoder_config"], batch=init_batch
        )
    else:
        fixed_encoder = None
    if config["encode_data"]:
        data_encoder = fixed_encoder
    else:
        data_encoder = None

    # Load data and encode if necessary
    print(config["train_data_dir"])
    train_dataset, init_batch = _load_data(
        dict(config, replay_dir=config["train_data_dir"]), data_encoder
    )
    eval_dataset, _ = _load_data(
        dict(config, replay_dir=config["eval_data_dir"]), data_encoder
    )

    if config["fix_eval_task"]:  # Ensure task id taken from data
        task_id = int(init_batch.observation["task_id"].flatten()[0])
    else:
        task_id = None

    # Load eval env
    eval_env = envs.registry.make(
        config["eval_env_name"],
        init_seed=config["seed"],
        task_id=task_id,
    )

    # Build agent
    learner = algs.registry.make(
        config["agent_name"],
        config=config,
        batch=init_batch,
        fixed_encoder=fixed_encoder,
    )

    # Setup logging
    log_dir = training_utils.make_path(config["project_dir"] / "logs", config)
    logger = training_utils.Logger(
        config, log_dir, config["agent_name"], use_wandb=not test
    )

    # Train
    time_start = time.time()
    for step in range(1, config["num_steps"] + 1):
        if (
            step % config["rollout_freq"] == 0
            or step == 1
            and config.get("rollout_on_start", False)
        ):
            if config.get("condition_on_context", False):
                batch = next(train_dataset)
            info = env_eval.evaluate(eval_env, learner, config["num_rollouts"])
            logger.write(info, "eval", step)

        batch = next(train_dataset)
        info = learner.update(batch)

        if step % config["log_freq"] == 0:
            info.update({"steps_per_second": step / (time.time() - time_start)})
            logger.write(info, "train", step)

        if step % config["eval_freq"] == 0:
            batch = next(eval_dataset)
            info = learner.eval(batch)
            logger.write(info, "val", step)

        if step < 100:  # Debugging purposes
            print(info)
    logger.close()

    # Store model
    ckpt_path = training_utils.make_path(config["project_dir"] / "ckpts", config)
    learner.save_checkpoint(ckpt_path, step)

    # Test load model
    learner.restore_checkpoint(ckpt_path)
