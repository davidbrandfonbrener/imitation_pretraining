"""Wrappers for DM control environments.

based on:
https://github.com/facebookresearch/drqv2/blob/main/dmc.py"""
from typing import Any, NamedTuple, Optional

import dm_env
import numpy as np

from dm_control.suite.wrappers import pixels as pixel_wrapper
from dm_control.suite.wrappers import action_scale as action_scale_wrapper
from dm_env import StepType, specs

# pylint: disable=missing-docstring


def add_wrappers(
    env: dm_env.Environment,
    include_pixels: bool = True,
    frame_size: int = 84,
    include_robot_state: bool = True,
    include_goal: bool = True,
    action_dtype: type = np.float32,
    action_repeat: Optional[int] = None,
    state_dtype: type = np.float32,
    task_id: Optional[int] = None,
    terminal_discount_zero: bool = True,
):
    # Action wrappers.
    env = ActionDTypeWrapper(env, action_dtype)
    env = action_scale_wrapper.Wrapper(env, minimum=-1.0, maximum=+1.0)
    if action_repeat is not None:
        env = ActionRepeatWrapper(env, action_repeat)

    # Observation wrappers.
    env = StateDTypeWrapper(env, state_dtype)
    if not include_goal:
        env = RemoveGoalWrapper(env)
    if not include_robot_state:
        env = RemoveRobotStateWrapper(env)
    if include_pixels:
        render_kwargs = dict(height=frame_size, width=frame_size, camera_id=0)
        env = pixel_wrapper.Wrapper(
            env,
            pixels_only=False,
            render_kwargs=render_kwargs,
            observation_key="pixels",
        )
    if terminal_discount_zero:
        env = LastDiscountWrapper(env)

    if task_id is None:  # Add dummy task id if not provided.
        task_id = -1
    env = TaskIDWrapper(env, task_id=task_id)

    # Return extended timestep.
    env = ExtendedTimeStepWrapper(env)
    return env


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class StateDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_obs_spec = env.observation_spec()
        for key in wrapped_obs_spec:
            if key == "pixels":
                wrapped_obs_spec[key] = specs.Array(
                    wrapped_obs_spec[key].shape,
                    np.uint8,
                    key,
                )
            else:
                wrapped_obs_spec[key] = specs.Array(
                    wrapped_obs_spec[key].shape,
                    dtype,
                    key,
                )
        self._obs_spec = wrapped_obs_spec

    def _cast_observation(self, time_step):
        obs = time_step.observation
        for key in obs:
            obs[key] = obs[key].astype(self._obs_spec[key].dtype)
        return time_step._replace(observation=obs)

    def step(self, action):
        return self._cast_observation(self._env.step(action))

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._cast_observation(self._env.reset())

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=np.float32(time_step.reward) or np.float32(0.0),
            discount=np.float32(time_step.discount) or np.float32(1.0),
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RemoveGoalWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._remove_goal(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._remove_goal(time_step)

    def _remove_goal(self, time_step):
        obs = time_step.observation
        if "goal" in obs:
            del obs["goal"]
        return time_step._replace(observation=obs)

    def observation_spec(self):
        obs_spec = self._env.observation_spec()
        if "goal" in obs_spec:
            del obs_spec["goal"]
        return obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RemoveRobotStateWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._remove_robot_state(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._remove_robot_state(time_step)

    def _remove_robot_state(self, time_step):
        obs = time_step.observation
        for key in obs:
            if key != "pixels":
                del obs[key]
        return time_step._replace(observation=obs)

    def observation_spec(self):
        obs_spec = self._env.observation_spec()
        for key in obs_spec:
            if key != "pixels":
                del obs_spec[key]
        return obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class TaskIDWrapper(dm_env.Environment):
    def __init__(self, env, task_id):
        self._env = env
        self._task_id = task_id

    def reset(self):
        time_step = self._env.reset()
        return self._augment_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_observation(time_step)

    def _augment_observation(self, time_step):
        obs = time_step.observation
        obs["task_id"] = np.array([self._task_id], dtype=np.int64)
        return time_step._replace(observation=obs)

    def observation_spec(self):
        obs_spec = self._env.observation_spec()
        obs_spec["task_id"] = specs.Array((1,), np.int64, "task_id")
        return obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class LastDiscountWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        if time_step.last():
            time_step = time_step._replace(discount=np.float32(0.0))
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
