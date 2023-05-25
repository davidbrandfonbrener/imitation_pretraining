"""A wrapper to turn metaworld environments into dm_env environments."""
from collections import OrderedDict
import copy
import numpy as np
import dm_env
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaworldEnv(dm_env.Environment):
    """A wrapper to turn metaworld environments into dm_env environments."""

    def __init__(
        self,
        task_name,
        task_seed,
        init_seed=None,
        resolution=120,
        goal_in_image=True,
        noise_scale=0.0,
    ):
        cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_name}-goal-observable"]
        self.task_name = task_name
        self._env = cls(seed=task_seed)  # Task seed sets the goal and object pos
        self._env.max_path_length = 250  # Shorten maximum trajectory length
        self._resolution = resolution
        self._goal_in_image = goal_in_image
        self._noise_scale = noise_scale
        # Init seed determines if hand is randomly initialized
        self._random_init = False if init_seed is None else True
        if self._random_init:
            self._init_rng = np.random.default_rng(seed=init_seed)
            self._default_hand_pos = self._env.hand_init_pos

        self._obs = self.reset().observation

    def _obs_to_dict(self, obs, info):
        if not self._goal_in_image:  # Hide the goal in the image
            goal_names = [n for n in self._env.model.site_names if "goal" in n]
            for name in goal_names:
                goal_idx = self._env.model.site_name2id(name)
                self._env.model.site_rgba[goal_idx, 3] = 0

        if self._noise_scale > 0.0:  # Add noise to the observations
            state = self._env.sim.get_state()
            new_state = copy.deepcopy(state)
            pos_len = new_state.qpos.shape[0]
            noise = self._env.np_random.randn(pos_len)
            new_state.qpos[:pos_len] += self._noise_scale * noise
            self._env.sim.set_state(new_state)
            self._env.sim.forward()

        res = int(self._resolution / 3)  # Used to crop the image
        pixels = self._env.render(
            offscreen=True,
            camera_name="corner2",
            resolution=(8 * res, 6 * res),
        )
        pixels = pixels[2 * res : 2 * res + 3 * res, 2 * res : 2 * res + 3 * res]

        if self._noise_scale > 0.0:  # Restore original sim state
            self._env.sim.set_state(state)
            self._env.sim.forward()
            assert np.allclose(self._env.sim.data.qpos, state.qpos, atol=1e-6)
            assert np.allclose(self._env.sim.data.qvel, state.qvel, atol=1e-6)

        return OrderedDict(
            {
                "position": obs[:4],
                "object": obs[4:18],
                "position_past": obs[18:22],
                "object_past": obs[22:36],
                "goal": obs[36:],
                "pixels": pixels,
                "success": np.array([info.get("success", 0.0)], dtype=np.float32),
            }
        )

    def _init_hand(self):
        """Randomly initialize the hand position."""
        default = self._default_hand_pos
        offset = np.array([0.05, 0.05, 0.05])
        init_pos = self._init_rng.uniform(low=default - offset, high=default + offset)
        self._env.hand_init_pos = init_pos

    def reset(self):
        if self._random_init:
            self._init_hand()
        self._env.reset()
        # Random action to ensure that goal is properly reset
        obs, _, _, _ = self._env.step(self._env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        obs = self._obs_to_dict(obs, {})
        return dm_env.restart(obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._obs_to_dict(obs, info)
        success = info.get("success", 0.0)  # terminate on success
        if done or self._env.curr_path_length == self._env.max_path_length or success:
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

    def __getattr__(self, name):
        return getattr(self._env, name)


class MetaworldBenchmarkEnv(MetaworldEnv):
    """A wrapper to make train-test splits for metworld envs."""

    def __init__(
        self,
        pretrain=True,
        split_key=0,
        eval_task_id=0,  # Task id for finetuning in range(5)
        task_seed=0,
        init_seed=None,
        resolution=120,
        goal_in_image=True,
        ntasks=None,
    ):
        self.pretrain = pretrain
        self.split_key = split_key
        if pretrain:
            self.env_list = SPLITS[split_key]["pretrain"]
            rng = np.random.default_rng(seed=task_seed)
            if ntasks is not None:  # Sample a subset of tasks
                self.env_list = rng.choice(self.env_list, size=ntasks, replace=False)
            self.task = rng.choice(self.env_list)
        else:
            self.env_list = SPLITS[split_key]["finetune"]
            self.task = self.env_list[eval_task_id]

        super().__init__(
            self.task,
            task_seed=task_seed,
            init_seed=init_seed,
            resolution=resolution,
            goal_in_image=goal_in_image,
        )


def dict_to_raw_obs(obs_dict):
    "Convert a dict observation to a raw observation, needed for scripted policies."
    return np.concatenate(
        [
            obs_dict["position"],
            obs_dict["object"],
            obs_dict["position_past"],
            obs_dict["object_past"],
            obs_dict["goal"],
        ]
    )


ENV_LIST = [
    # Split 0 test envs (aka original ML45 test envs)
    "bin-picking-v2",
    "box-close-v2",
    "hand-insert-v2",
    "door-lock-v2",
    "door-unlock-v2",
    # Split 1 test envs
    "assembly-v2",
    "basketball-v2",
    "button-press-topdown-v2",
    "button-press-topdown-wall-v2",
    "button-press-v2",
    # Split 2 test envs
    "button-press-wall-v2",
    "coffee-button-v2",
    "coffee-pull-v2",
    "coffee-push-v2",
    "dial-turn-v2",
    # Split 3 test envs
    "disassemble-v2",
    "door-close-v2",
    "door-open-v2",
    "drawer-close-v2",
    "drawer-open-v2",
    # Split 4 test envs
    "faucet-open-v2",
    "faucet-close-v2",
    "hammer-v2",
    "handle-press-side-v2",
    "handle-press-v2",
    # Split 5 test envs
    "handle-pull-side-v2",
    "handle-pull-v2",
    "lever-pull-v2",
    "peg-insert-side-v2",
    "pick-place-wall-v2",
    # Split 6 test envs
    "pick-out-of-hole-v2",
    "reach-v2",
    "push-back-v2",
    "push-v2",
    "pick-place-v2",
    # Split 7 test envs
    "plate-slide-v2",
    "plate-slide-side-v2",
    "plate-slide-back-v2",
    "plate-slide-back-side-v2",
    "peg-insert-side-v2",
    # Split 8 test envs
    "peg-unplug-side-v2",
    "soccer-v2",
    "stick-push-v2",
    "stick-pull-v2",
    "push-wall-v2",
    # Split 9 test envs
    "push-v2",
    "reach-wall-v2",
    "reach-v2",
    "shelf-place-v2",
    "sweep-into-v2",
    # Extra envs
    "sweep-v2",
    "window-open-v2",
    "window-close-v2",
]

R3M_LIST = [
    "assembly-v2",
    "bin-picking-v2",
    "button-press-v2",
    "drawer-open-v2",
    "hammer-v2",
]

SPLITS = {
    "0": {"pretrain": ENV_LIST[5:], "finetune": ENV_LIST[:5]},
    "1": {"pretrain": ENV_LIST[:5] + ENV_LIST[10:], "finetune": ENV_LIST[5:10]},
    "2": {"pretrain": ENV_LIST[:10] + ENV_LIST[15:], "finetune": ENV_LIST[10:15]},
    # R3M split
    "r3m": {
        "pretrain": [e for e in ENV_LIST if e not in R3M_LIST],
        "finetune": R3M_LIST,
    },
    # Split with all data for pretraining
    "all": {"pretrain": ENV_LIST, "finetune": ENV_LIST[:5]},
    # Special hand-designed splits
    "plate": {
        "pretrain": ["plate-slide-v2", "plate-slide-side-v2", "plate-slide-back-v2"],
        "finetune": ["plate-slide-back-side-v2"],
    },
    "door": {
        "pretrain": ["door-unlock-v2", "door-close-v2", "door-open-v2"],
        "finetune": ["door-lock-v2"],
    },
    "button": {
        "pretrain": [
            "button-press-topdown-v2",
            "button-press-topdown-wall-v2",
            "button-press-wall-v2",
        ],
        "finetune": ["button-press-v2"],
    },
    "plate-all": {
        "pretrain": [
            "plate-slide-v2",
            "plate-slide-side-v2",
            "plate-slide-back-v2",
            "plate-slide-back-side-v2",
        ],
        "finetune": ["plate-slide-back-side-v2"],
    },
    "door-all": {
        "pretrain": [
            "door-unlock-v2",
            "door-close-v2",
            "door-open-v2",
            "door-lock-v2",
        ],
        "finetune": ["door-lock-v2"],
    },
    "button-all": {
        "pretrain": [
            "button-press-topdown-v2",
            "button-press-topdown-wall-v2",
            "button-press-wall-v2",
            "button-press-v2",
        ],
        "finetune": ["button-press-v2"],
    },
}
