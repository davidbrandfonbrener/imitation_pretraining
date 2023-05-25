"""Convert kitchen data to replay dataset."""
import shutil
import numpy as np
from dm_env import specs
from imitation_pretraining import envs
from imitation_pretraining.data_utils import replay_buffer

KITCHEN_PATH = "PATH TO DATA"

# Note: several of the trajectories are mislabeled in the dataset
# We remove them based on their index
BAD_TRAJS = [77, 106, 133, 158, 322, 364, 365, 391, 414, 493, 511, 540, 546]


class KitchenGenerator:
    """Class to convert kitchen dataset to replay buffers"""

    def __init__(self, config) -> None:
        self.config = config

        pretrain_name = "pretrain" if config["pretrain"] else "finetune"
        replay_dir = config["root_path"] / "data" / config["env_name"] / pretrain_name
        filename = f"ep-{config['train_eps']}-seed-{config['seed']}"
        if config.get("use_all_data", False):
            filename += f"-all_data"
        replay_path = replay_dir / filename
        self.replay_path = replay_path

        # Multiproces-safe clean dir
        leader = "worker_id" not in config or config["worker_id"] == 0
        if leader:
            shutil.rmtree(replay_path, ignore_errors=True)

        self.split = int(config["env_name"].split("_")[-1])
        assert self.split in [0, 1, 2]

        self.env = envs.registry.make(config["env_name"], init_seed=0, task_id=0)
        data_specs = {
            "observation": self.env.observation_spec(),
            "action": self.env.action_spec(),
            "reward": specs.Array((1,), np.float32, "reward"),
            "discount": specs.Array((1,), np.float32, "discount"),
            "step_type": specs.Array((1,), np.int32, "step_type"),
        }
        self.buffer_storage = replay_buffer.ReplayBufferStorage(
            data_specs, replay_path, clean_dir=False, write_json=leader
        )

        # Make eval replay buffer
        eval_filename = f"{filename}-eval"
        eval_replay_path = replay_dir / eval_filename
        self.eval_buffer_storage = replay_buffer.ReplayBufferStorage(
            data_specs, eval_replay_path, clean_dir=False, write_json=leader
        )

    def generate(self):
        """Translate data to replay buffer."""
        # Load dataset
        data_dir = KITCHEN_PATH

        actions_seq = np.load(f"{data_dir}/actions_seq.npy")
        observations_seq = np.load(f"{data_dir}/observations_seq.npy")
        completed_task_ids = np.load(f"{data_dir}/completed_task_ids.npy")
        onehot_goals = np.stack(
            [np.concatenate([np.eye(7)[j] for j in c]) for c in completed_task_ids],
        )
        existence_mask = np.load(f"{data_dir}/existence_mask.npy")
        completion_times = [
            np.max(np.where(existence_mask[:, i] == 1)) + 1
            for i in range(existence_mask.shape[1])
        ]
        max_time, num_episodes, _ = actions_seq.shape

        # Cast types
        actions_seq = actions_seq.astype(np.float32)
        observations_seq = observations_seq.astype(np.float32)
        onehot_goals = onehot_goals.astype(np.float32)

        # Select episodes for split.
        if self.config["pretrain"]:
            if self.config.get("use_all_data", None) is not None:  # Select all episodes
                ep_ids = list(range(len(completed_task_ids)))

            else:  # Select all episodes that do not complete the target tasks
                splits = envs.kitchen.ENV_SPLITS
                target_ids = []
                for split in splits:
                    target_tasks = split().TASK_ELEMENTS
                    target_id = [envs.kitchen.ALL_TASKS.index(t) for t in target_tasks]
                    target_ids.append(target_id)
                ep_ids = []
                for i in range(num_episodes):
                    i_completes_target = False
                    for target_id in target_ids:
                        if np.sum(np.abs(target_id - completed_task_ids[i])) == 0:
                            i_completes_target = True
                    if not i_completes_target:
                        ep_ids.append(i)

        else:  # Only use episodes that complete the target task
            target_tasks = envs.kitchen.ENV_SPLITS[self.split]().TASK_ELEMENTS
            target_id = [envs.kitchen.ALL_TASKS.index(t) for t in target_tasks]
            ep_ids = np.array(
                [
                    i
                    for i in range(num_episodes)
                    if np.sum(np.abs(target_id - completed_task_ids[i])) == 0
                ]
            )

        # Remove bad trajectories
        ep_ids = np.array([i for i in ep_ids if i not in BAD_TRAJS])

        # Randomly split into train and eval using seed
        np.random.seed(self.config["seed"])
        train_ids = np.random.choice(ep_ids, self.config["train_eps"], replace=False)
        eval_ids = np.array([i for i in ep_ids if i not in train_ids], dtype=np.int32)
        eval_ids = eval_ids[: min(len(eval_ids), self.config["eval_eps"])]

        # Assemble trajectories
        def idx_to_trajs(idx):
            trajs = []
            for i in idx:
                actions = actions_seq[: completion_times[i], i]
                obses = observations_seq[: completion_times[i], i, :30]
                goal = onehot_goals[i]
                trajs.append({"observations": obses, "actions": actions, "goal": goal})
            return trajs

        train_trajs = idx_to_trajs(train_ids)
        eval_trajs = idx_to_trajs(eval_ids)

        if "worker_id" in self.config:
            train_trajs = train_trajs[
                self.config["worker_id"] :: self.config["n_workers"]
            ]
            eval_trajs = eval_trajs[
                self.config["worker_id"] :: self.config["n_workers"]
            ]

        # Store episodes in respective replay buffers
        for i, traj in enumerate(train_trajs):
            self.store_episode(traj, self.buffer_storage, train_ids[i])
        for i, traj in enumerate(eval_trajs):
            self.store_episode(traj, self.eval_buffer_storage, eval_ids[i])

    def store_episode(self, episode, buffer_storage, task_id):
        """Store episode in replay buffer."""

        goal = episode["goal"]

        # First timestep
        obs = episode["observations"][0]
        timestep = self.env.data_to_dm(
            obs, None, goal, task_id, terminal=False, initial=True
        )
        buffer_storage.add(timestep)

        # Middle timesteps
        for i in range(1, len(episode["observations"]) - 1):
            obs = episode["observations"][i]
            action = episode["actions"][i - 1]

            timestep = self.env.data_to_dm(
                obs, action, goal, task_id, terminal=False, initial=False
            )
            buffer_storage.add(timestep)

        # Last timestep
        obs = episode["observations"][-1]
        action = episode["actions"][-2]
        timestep = self.env.data_to_dm(
            obs, action, goal, task_id, terminal=True, initial=False
        )
        buffer_storage.add(timestep)
