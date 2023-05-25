"""Wrapper around D4RL kitchen environment."""
from collections import OrderedDict
import copy
import dm_env
import numpy as np

from d4rl.kitchen.kitchen_envs import KitchenBase
from dm_control.mujoco import engine
from imitation_pretraining.envs.wrappers import ExtendedTimeStep


class KitchenSplit0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "slide cabinet"]


class KitchenSplit1(KitchenBase):
    TASK_ELEMENTS = ["bottom burner", "top burner", "slide cabinet", "hinge cabinet"]


class KitchenSplit2(KitchenBase):
    TASK_ELEMENTS = ["kettle", "bottom burner", "top burner", "light switch"]


ENV_SPLITS = [KitchenSplit0, KitchenSplit1, KitchenSplit2]

ALL_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]


def get_onehot_goal(task_elements):
    task_ids = [ALL_TASKS.index(e) for e in task_elements]
    return np.concatenate([np.eye(7)[i] for i in task_ids])


class KitchenEnv(dm_env.Environment):
    """Wrapper around D4RL kitchen environment."""

    def __init__(
        self,
        split=0,
        init_seed=None,
        resolution=120,
        goal_in_obs=False,
        pixels_in_obs=False,
    ) -> None:
        self._resolution = resolution
        self._goal_in_obs = goal_in_obs
        self._pixels_in_obs = pixels_in_obs

        self._env = ENV_SPLITS[split]()
        self._env.seed(init_seed)
        np_obs = self._env.reset()
        self._obs = self._get_obs(np_obs)

        self._max_episode_steps = 280
        self._step = 0

    def _get_obs(self, np_obs, success=0.0):
        obs_dict = OrderedDict(
            {
                "position": np_obs[:9],
                "object": np_obs[9:30],
                "goal": get_onehot_goal(self._env.TASK_ELEMENTS),
                "success": np.array([success], dtype=np.float32),
            }
        )
        if self._pixels_in_obs:
            obs_dict["pixels"] = self.render(np_obs[:30])
        return obs_dict

    def reset(self):
        self._step = 0
        np_obs = self._env.reset()
        obs = self._get_obs(np_obs)
        return dm_env.restart(obs)

    def step(self, action):
        self._step += 1
        np_obs, reward, done, info = self._env.step(action)
        assert "images" not in info.keys()
        success = int(len(self._env.tasks_to_complete) == 0)
        obs = self._get_obs(np_obs, success=success)
        if done or self._step >= self._max_episode_steps:
            return dm_env.termination(reward, obs)
        else:
            return dm_env.transition(reward, obs)

    def observation_spec(self):
        spec = OrderedDict(
            {
                k: dm_env.specs.Array(
                    shape=self._obs[k].shape,
                    dtype=self._obs[k].dtype,
                )
                for k in self._obs.keys()
            }
        )
        return spec

    def action_spec(self):
        space = self._env.action_space
        spec = dm_env.specs.BoundedArray(
            shape=space.shape, dtype=space.dtype, minimum=space.low, maximum=space.high
        )
        return spec

    def render(self, qpos=None) -> np.ndarray:
        """Render qpos image from camera view."""
        if qpos is not None:  # Temporarily override sim state
            data = copy.deepcopy(self._env.sim.data)
            self._env.sim.data.qpos = qpos
            self._env.sim.forward()

        camera = engine.MovableCamera(self._env.sim, self._resolution, self._resolution)
        camera.set_pose(
            distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
        )
        im = camera.render()

        if qpos is not None:  # Restore original sim state
            self._env.sim.data.qpos = data.qpos
            self._env.sim.data.qvel = data.qvel
            self._env.sim.forward()
            assert np.allclose(self._env.sim.data.qpos, data.qpos, atol=1e-6)
            assert np.allclose(self._env.sim.data.qvel, data.qvel, atol=1e-6)
        return im

    def __getattr__(self, name):
        return getattr(self._env, name)

    def data_to_dm(self, obs, action, goal, info, terminal, initial):
        """Convert raw observation to dm_env extended timestep."""
        qp = obs[:9]
        obj_qp = obs[9:30]
        obs_dict = OrderedDict(
            {
                "position": qp,
                "object": obj_qp,
                "goal": goal,
                "success": np.array([0.0], dtype=np.float32),  # dummy variable
            }
        )
        if self._pixels_in_obs:
            # Just set sim state directly for simplicity
            self._env.sim.data.qpos[:9] = qp
            self._env.sim.data.qpos[9:] = obj_qp
            self._env.sim.forward()
            obs_dict["pixels"] = self.render(None)

        # Add task id last to maintain ordering
        obs_dict["task_id"] = np.array([info], dtype=np.int64)

        if initial:
            timestep = dm_env.restart(obs_dict)
        elif terminal:
            timestep = dm_env.termination(None, obs_dict)
        else:
            timestep = dm_env.transition(None, obs_dict)

        if action is None:  # Dummy action for initial timestep
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=timestep.observation,
            step_type=timestep.step_type,
            action=action,
            reward=np.float32(timestep.reward) or np.float32(0.0),
            discount=np.float32(timestep.discount) or np.float32(1.0),
        )
