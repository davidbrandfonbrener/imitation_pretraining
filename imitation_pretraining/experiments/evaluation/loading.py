"""Load agent from checkpoint"""
import os
import pickle
from typing import Callable, Dict, List, Tuple
from imitation_pretraining import algs
from imitation_pretraining.data_utils import replay_buffer, Batch


def get_batch(replay_dir: str, encoder: Callable, history: int, nstep: int) -> Batch:
    """Get batch from replay buffer

    Args:
        replay_dir (str): path to replay buffer
        encoder (Encoder): encoder to use
        history (int): number of frames in history
        nstep (int): nstep

    Returns:
        Batch: batch
    """
    buffer = replay_buffer.ReplayBufferDataset(
        replay_dir=replay_dir,
        batch_size=100,
        max_episodes=5,
        nstep=nstep,
        history=history,
        encoder=encoder,
    )
    return next(iter(buffer))


def load_sweep_configs(root_path: str, date: str, sweep_id: int) -> Tuple[List]:
    """Load sweep list from checkpoint

    Args:
        root_path (str): path from root directory to parent dir of logs and ckpts
        date (str): date of experiment
        sweep_id (int): slurm sweep id of experiment

    Returns:
        list: list of sweep config dicts
    """
    log_path = os.path.join(root_path, "logs", date, str(sweep_id))
    contents = [os.path.join(log_path, name) for name in os.listdir(log_path)]
    subdirs = [name for name in contents if os.path.isdir(name)]
    sweep_config_list = []
    for job_path in subdirs:
        try:
            with open(os.path.join(job_path, "config.pkl"), "rb") as f:
                config = pickle.load(f)
            sweep_config_list.append(config)
        except:
            print(f"Could not load config from {job_path}")
    return sweep_config_list


def load_agent(root_path: str, date: str, sweep_id: int, job_id: int) -> Dict:
    """Load agent from checkpoint

    Args:
        root_path (str): path from root directory to parent dir of logs and ckpts
        date (str): date of experiment
        sweep_id (int): slurm sweep id of experiment
        job_id (int): slurm job id within experiment

    Returns:
        dict: agent, encoder, config, batch
    """
    # Load config
    log_path = os.path.join(root_path, "logs", date, str(sweep_id), str(job_id))
    with open(os.path.join(log_path, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    # Load encoder if separate from agent:
    if config["encoder_name"] is not None:
        enc_config = config["encoder_config"]
        batch = get_batch(
            config["train_data_dir"], None, enc_config["history"], enc_config["nstep"]
        )
        # This make call will automatically load the encoder weights
        encoder = algs.registry.make(
            config["encoder_name"], config=config["encoder_config"], batch=batch
        )
    else:
        encoder = None

    # Load agent
    batch = get_batch(
        config["train_data_dir"], encoder, config["history"], config["nstep"]
    )
    agent = algs.registry.make(
        config["agent_name"],
        config=config,
        batch=batch,
        fixed_encoder=encoder,
    )
    ckpt_path = os.path.join(root_path, "ckpts", date, sweep_id, job_id)
    agent.restore_checkpoint(ckpt_path)

    return {"agent": agent, "encoder": encoder, "config": config, "batch": batch}
