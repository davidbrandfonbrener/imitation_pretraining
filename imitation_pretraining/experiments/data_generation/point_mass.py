"""Generate data for the goal-based point mass task."""
import functools as ft
import numpy as np

from imitation_pretraining import envs
from imitation_pretraining.experiments.data_generation import common


def demo_policy(timestep, sigma):
    """A policy that navigates to the goal"""
    k_p = 10.0
    k_d = 1.0
    raw_action = np.float32(
        k_p * (timestep.observation["goal"] - timestep.observation["position"])
        - k_d * timestep.observation["velocity"]
        + sigma * np.random.normal(size=(2,))
    )
    return np.clip(raw_action, -1.0, 1.0)


def build_policy(policy_type):
    """Converts policy type to policy."""
    if policy_type == "expert":
        policy = ft.partial(demo_policy, sigma=0.0)
    elif policy_type == "noisy":
        policy = ft.partial(demo_policy, sigma=2.0)
    else:
        raise ValueError(f"Invalid policy type: {policy_type}")
    return policy


class PointMassGenerator(common.DataGenerator):
    """Generates data for the point mass task."""

    def build_env_and_policy(self, task_id, rng):
        policy = build_policy(self.config["policy_type"])

        init_seed = rng.integers(np.iinfo(np.int32).max, dtype=int)
        env = envs.registry.make(
            self.config["env_name"],
            init_seed=init_seed,
            task_id=task_id,
        )
        return env, policy
