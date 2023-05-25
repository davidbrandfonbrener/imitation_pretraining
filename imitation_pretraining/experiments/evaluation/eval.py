"""Evaluation loop."""
from typing import Optional
import time
import dm_env
import numpy as np
from imitation_pretraining.algs.base import BaseLearner


def evaluate(env: dm_env.Environment, policy: BaseLearner, num_episodes: int) -> float:
    """Evaluate the policy."""
    outputs = []
    for _ in range(num_episodes):
        cum_reward = 0
        cum_success = 0
        start_time = time.time()
        timestep = env.reset()
        step = 0
        while not timestep.last():
            action = policy.sample_action(timestep.observation)
            timestep = env.step(action)
            cum_reward += timestep.reward
            if "success" in timestep.observation:
                cum_success += timestep.observation["success"]
            step += 1
        end_time = time.time()
        print(f"Eval return: {cum_reward:.2f}, time: {end_time - start_time:.2f}s")
        outputs.append(
            {
                "eval_return": cum_reward,
                "eval_success": int(cum_success > 0),
                "eval_steps_per_second": step / (end_time - start_time),
                "eval_ep_len": step,
            }
        )
    mean_outputs = {k: np.mean([o[k] for o in outputs]) for k in outputs[0]}
    mean_outputs["eval_return_std"] = np.std([o["eval_return"] for o in outputs])
    return mean_outputs
