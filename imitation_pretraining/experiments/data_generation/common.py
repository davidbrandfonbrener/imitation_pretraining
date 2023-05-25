"""Generator to build datasets."""
from dm_env import specs
import numpy as np
from imitation_pretraining import envs
from imitation_pretraining.data_utils import replay_buffer


def add_episode(env, policy, buffer_storage, success_only=False, max_retries=5):
    """Adds an episode to the replay buffer."""
    for _ in range(max_retries):
        ts_buffer = []
        timestep = env.reset()
        print("goal: ", timestep.observation["goal"])
        print("position: ", timestep.observation["position"])
        ts_buffer.append(timestep)
        while not timestep.last():
            action = policy(timestep)
            next_timestep = env.step(action)
            ts_buffer.append(next_timestep)
            timestep = next_timestep
        print("episode length: ", len(ts_buffer))

        success = ts_buffer[-1].observation["success"]
        if not success_only or success:
            for timestep in ts_buffer:
                buffer_storage.add(timestep)
            return True

    print(f"Failed on all {max_retries} attempts.")
    return False


class DataGenerator:
    """Class to generate data"""

    def __init__(self, config) -> None:
        self.config = config

        # Set up replay buffer
        filename = f"ep-{config['episodes']}-per-{config['episodes_per_seed']}"
        filename += f"-seed-{config['seed']}-{config['policy_type']}"
        pretrain_name = "pretrain" if config["pretrain"] else "finetune"
        replay_dir = config["root_path"] / "data" / config["env_name"] / pretrain_name
        replay_path = replay_dir / filename
        self.replay_path = replay_path
        print("Replay path: ", str(replay_path))

        env = envs.registry.make(config["env_name"], init_seed=0, task_id=0)
        data_specs = {
            "observation": env.observation_spec(),
            "action": env.action_spec(),
            "reward": specs.Array((1,), np.float32, "reward"),
            "discount": specs.Array((1,), np.float32, "discount"),
            "step_type": specs.Array((1,), np.int32, "step_type"),
        }
        self.buffer_storage = replay_buffer.ReplayBufferStorage(
            data_specs, replay_path, clean_dir=config["clean_dir"]
        )

        # Make eval replay buffer
        eval_filename = f"{filename}-eval"
        eval_replay_path = replay_dir / eval_filename
        self.eval_buffer_storage = replay_buffer.ReplayBufferStorage(
            data_specs, eval_replay_path, clean_dir=config["clean_dir"]
        )

    def build_env_and_policy(self, task_id, rng):
        """Builds the environment and policy."""
        raise NotImplementedError

    def generate(self):
        """Collect data."""
        seed = self.config["seed"]
        rng = np.random.default_rng(seed)
        num_task_seeds = self.config["episodes"] // self.config["episodes_per_seed"]
        task_ids = []
        for _ in range(num_task_seeds):
            task_id = rng.integers(np.iinfo(np.int32).max, dtype=int)
            task_ids.append(task_id)
            print("task_id: ", task_id)
            env, policy = self.build_env_and_policy(task_id, rng)
            for _ in range(self.config["episodes_per_seed"]):
                add_episode(
                    env, policy, self.buffer_storage, self.config["success_only"]
                )

        for task_id in task_ids[: min(20, len(task_ids))]:
            env, policy = self.build_env_and_policy(task_id, rng)
            for _ in range(max(self.config["episodes_per_seed"] // 2, 1)):
                add_episode(
                    env, policy, self.eval_buffer_storage, self.config["success_only"]
                )

        self.test_load()

    def test_load(self):
        """Test that the saved buffer loads properly"""
        buffer_loader = replay_buffer.ReplayBufferDataset(
            replay_dir=self.replay_path,
            max_episodes=1,
            nstep=1,
            discount=1.0,
            batch_size=5,
            seed=0,
            shuffle_and_repeat=True,
        )
        replay_iter = iter(buffer_loader)
        batch = next(replay_iter)
        print(batch._asdict().keys())
        print(batch.observation.keys())
        print(batch.observation["goal"].shape)
