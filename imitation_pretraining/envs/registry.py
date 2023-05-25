"""Environment registry."""
import functools as ft
from typing import Optional

from imitation_pretraining.envs import wrappers
from imitation_pretraining.envs import point_mass
from imitation_pretraining.envs import metaworld
from imitation_pretraining.envs import kitchen


class Registry(object):
    """A registry for environments."""

    def __init__(self):
        self._envs = {}

    def register(self, name, env_constructor_fn):
        """Register an environment."""
        self._envs[name] = env_constructor_fn
        return

    def make(self, name, **kwargs):
        """Build an environment. Return the env object."""
        return self._envs[name](**kwargs)


registry = Registry()


def build_point_mass_env(
    frame_size: int = 84,
    init_seed: int = 0,
    task_id: int = 0,
    noise_scale: float = 0.0,
    time_limit: float = point_mass.DEFAULT_TIME_LIMIT,
):
    """Builds a goal-based point_mass environment."""

    env = point_mass.goal_based(
        time_limit=time_limit,
        random=init_seed,
        task_id=task_id,
        vis_goal=False,
        noise_scale=noise_scale,
    )

    env = wrappers.add_wrappers(
        env,
        include_pixels=True,
        frame_size=frame_size,
        include_robot_state=True,
        include_goal=True,
        task_id=task_id,
        terminal_discount_zero=True,
    )
    return env


def build_metaworld(
    task_name: str,
    frame_size: int = 120,
    init_seed: Optional[int] = None,
    task_id: Optional[int] = None,
    goal_in_image: bool = True,
    noise_scale: float = 0.0,
    ntasks: Optional[int] = None,
):
    """Builds a single task from metaworld."""
    if "benchmark" in task_name:
        name_list = task_name.split("_")
        split_key = name_list[-1]

        if len(name_list) == 2:
            pretrain = True
            eval_id = None
        elif len(name_list) == 3:
            pretrain = False
            eval_id = int(name_list[-2])

        env = metaworld.MetaworldBenchmarkEnv(
            split_key=split_key,
            pretrain=pretrain,
            eval_task_id=eval_id,
            task_seed=task_id,
            init_seed=init_seed,
            resolution=frame_size,
            goal_in_image=goal_in_image,
            ntasks=ntasks,
        )
    else:
        env = metaworld.MetaworldEnv(
            task_name=task_name,
            task_seed=task_id,
            resolution=frame_size,
            init_seed=init_seed,
            goal_in_image=goal_in_image,
            noise_scale=noise_scale,
        )

    env = wrappers.add_wrappers(
        env,
        include_pixels=False,  # pixels already in observation
        include_robot_state=True,
        include_goal=True,
        task_id=task_id,
    )
    return env


def build_kitchen(
    split: int,  # Defines target task + split
    frame_size: Optional[int] = None,
    init_seed: Optional[int] = None,
    task_id: Optional[int] = None,
):
    """Builds a single task from metaworld."""
    env = kitchen.KitchenEnv(
        split=split,
        init_seed=init_seed,
        pixels_in_obs=frame_size is not None,
        resolution=frame_size,
    )

    env = wrappers.add_wrappers(
        env,
        include_pixels=False,  # pixels already in observation
        include_robot_state=True,
        include_goal=True,
        task_id=task_id,
    )
    return env


### REGISTRY ###

registry.register("point_mass", build_point_mass_env)
registry.register(
    "point_mass_noisy", ft.partial(build_point_mass_env, noise_scale=0.01)
)


for eval_env in metaworld.ENV_LIST:
    env_name = "_".join(eval_env.split("-")[:-1])
    registry.register(
        f"metaworld_{env_name}",
        ft.partial(build_metaworld, task_name=eval_env, frame_size=120),
    )

    registry.register(
        f"metaworld_{env_name}_nogoal",
        ft.partial(
            build_metaworld, task_name=eval_env, frame_size=120, goal_in_image=False
        ),
    )
    registry.register(
        f"metaworld_{env_name}_noisy",
        ft.partial(
            build_metaworld, task_name=eval_env, frame_size=120, noise_scale=0.01
        ),
    )

    registry.register(
        f"metaworld_{env_name}_nogoal_noisy",
        ft.partial(
            build_metaworld,
            task_name=eval_env,
            frame_size=120,
            goal_in_image=False,
            noise_scale=0.01,
        ),
    )


for split in list(range(10)) + [
    "r3m",
    "all",
    "plate",
    "button",
    "door",
    "plate-all",
    "button-all",
    "door-all",
]:

    registry.register(
        f"metaworld_pretrain_split_{split}",
        ft.partial(build_metaworld, task_name=f"benchmark_{split}", frame_size=120),
    )
    for ntasks in [10, 20, 45]:
        registry.register(
            f"metaworld_pretrain_split_{split}_ntasks_{ntasks}",
            ft.partial(
                build_metaworld,
                task_name=f"benchmark_{split}",
                frame_size=120,
                ntasks=ntasks,
            ),
        )

    for eval_task_id in range(5):
        registry.register(
            f"metaworld_finetune_{eval_task_id}_split_{split}",
            ft.partial(
                build_metaworld,
                task_name=f"benchmark_{eval_task_id}_{split}",
                frame_size=120,
            ),
        )

for split in range(3):
    registry.register(
        f"kitchen_split_{split}",
        ft.partial(build_kitchen, split=split, frame_size=120),
    )

    registry.register(
        f"kitchen_split_{split}_state",
        ft.partial(build_kitchen, split=split, frame_size=None),
    )
