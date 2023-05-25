"""Generate data for metaworld tasks."""
import numpy as np
from metaworld import policies
from imitation_pretraining import envs
from imitation_pretraining.envs import metaworld
from imitation_pretraining.experiments.data_generation import common


def get_policy(task_name, std=0.0):
    """Get hand-coded policy from metaworld."""
    policy_name = "Sawyer"
    if task_name == "peg-insert-side-v2":
        policy_name += "PegInsertionSideV2"
    else:
        parse_name = task_name.split("-")
        for n in parse_name:
            policy_name += n.capitalize()
    policy_name += "Policy"
    policy = getattr(policies, policy_name)()

    def policy_wrapper(timestep):
        obs = metaworld.dict_to_raw_obs(timestep.observation)
        action = np.float32(policy.get_action(obs))
        if std > 0.0:
            noise = std * np.random.normal(size=4)
            noise[-1] = 0  # No noise on gripper
            action += noise
        return np.clip(action, -1.0, 1.0)

    return policy_wrapper


class MetaWorldGenerator(common.DataGenerator):
    """Generates data for the point mass task."""

    def build_env_and_policy(self, task_id, rng):
        init_seed = rng.integers(np.iinfo(np.int32).max, dtype=int)
        assert isinstance(init_seed, int)
        env = envs.registry.make(
            self.config["env_name"],
            task_id=task_id,
            init_seed=init_seed,
        )
        print(f"Built {env.task_name}")

        if self.config["policy_type"] == "expert":
            std = 0.0
        elif self.config["policy_type"] == "noisy":
            std = 0.5
        else:
            raise ValueError(f"Invalid policy type: {self.config['policy_type']}")

        policy = get_policy(env.task_name, std=std)
        return env, policy
