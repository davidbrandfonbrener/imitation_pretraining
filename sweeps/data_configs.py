"""Configs for datasets."""
from sweep_utils import grid_to_list


def finteune_data_configs(sweep_size: bool):
    """Configs for finetuning datasets.
    Note, each dataset also supports 5 seeds.
    18 or 72 datasets, depending on sweep_size."""
    config_list = []

    # Pointmass
    env_grid = dict(
        pretrain=[False],
        eval_env_name=["point_mass"],
        fix_eval_task=[True],
        ep=[10],
        per=[10],
        max_episodes=[1, 2, 5, 10] if sweep_size else [2],
    )
    config_list.extend(grid_to_list(env_grid))

    # Pick and place
    env_grid = dict(
        pretrain=[False],
        eval_env_name=["metaworld_pick_place_nogoal"],
        fix_eval_task=[True],
        ep=[20],
        per=[20],
        max_episodes=[2, 5, 10, 20] if sweep_size else [5],
    )
    config_list.extend(grid_to_list(env_grid))

    # Metaworld
    env_grid = dict(
        pretrain=[False],
        eval_env_name=[
            f"metaworld_finetune_0_split_{s}"
            for s in ["door"]  # ["button", "door", "plate"]
        ]
        + [f"metaworld_finetune_{i}_split_{s}" for s in ["0", "r3m"] for i in range(5)],
        fix_eval_task=[False],
        ep=[20],
        per=[1],
        max_episodes=[2, 5, 10, 20] if sweep_size else [10],
    )
    config_list.extend(grid_to_list(env_grid))

    # Kitchen
    env_grid = dict(
        pretrain=[False],
        eval_env_name=[f"kitchen_split_{i}" for i in range(3)],
        fix_eval_task=[False],
        ep=[15],
        per=[1],
        max_episodes=[2, 5, 10, 15] if sweep_size else [10],
    )
    config_list.extend(grid_to_list(env_grid))

    return config_list


def pretrain_data_configs(sweep_size: bool):
    """Configs for pretraining datasets.
    6 or 18 datasets, depending on sweep_size."""
    config_list = []

    # Pointmass + Pick and place
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=["point_mass", "metaworld_pick_place_nogoal"],
        fix_eval_task=[True],
        ep=[1000],
        per=[1],
        max_episodes=[10, 100, 1000] if sweep_size else [100],
    )
    config_list.extend(grid_to_list(env_grid))

    # Kitchen
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[f"kitchen_split_0"],
        fix_eval_task=[False],
        ep=[450],
        per=[1],
        max_episodes=[50, 150, 450] if sweep_size else [450],
    )
    config_list.extend(grid_to_list(env_grid))

    # Metaworld BDP
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[
            f"metaworld_pretrain_split_{s}" for s in ["door"]  # , "plate", "button"]
        ],
        fix_eval_task=[False],
        ep=[1000],
        per=[1],
        max_episodes=[10, 100, 1000] if sweep_size else [100],
    )
    config_list.extend(grid_to_list(env_grid))

    # Metaworld large
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[f"metaworld_pretrain_split_{s}" for s in ["0", "r3m"]],
        fix_eval_task=[False],
        ep=[10000],
        per=[1],
        max_episodes=[100, 1000, 10000] if sweep_size else [1000],
    )
    config_list.extend(grid_to_list(env_grid))

    return config_list


def pretrain_data_all_tasks_configs():
    """Configs for pretraining datasets including finetune tasks.
    5 datasets."""

    config_list = []

    # Kitchen with all data
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[f"kitchen_split_0"],
        all_data=[True],
        fix_eval_task=[False],
        ep=[450],
        per=[1],
        max_episodes=[450],
    )
    config_list.extend(grid_to_list(env_grid))

    # Metaworld BDP
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[
            f"metaworld_pretrain_split_{s}-all"
            for s in ["door"]  # , "plate", "button"]
        ],
        fix_eval_task=[False],
        ep=[1000],
        per=[1],
        max_episodes=[100],
    )
    config_list.extend(grid_to_list(env_grid))

    # Metaworld large
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[f"metaworld_pretrain_split_{s}" for s in ["all"]],
        fix_eval_task=[False],
        ep=[10000],
        per=[1],
        max_episodes=[10000],
    )
    config_list.extend(grid_to_list(env_grid))

    return config_list


def pretrain_data_diversity_configs():
    """Configs for pretraining datasets including finetune tasks.
    12 datasets."""

    config_list = []

    # Pointmass + Pick and place
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=["point_mass", "metaworld_pick_place_nogoal"],
        fix_eval_task=[True],
        ep=[100],
        per=[1, 10, 100],
        max_episodes=[100],
    )
    config_list.extend(grid_to_list(env_grid))

    # Metaworld large
    env_grid = dict(
        seed=[0],
        pretrain=[True],
        eval_env_name=[
            f"metaworld_pretrain_split_{s}_ntasks_{n}"
            for s in ["0", "r3m"]
            for n in [10, 20, 45]
        ],
        fix_eval_task=[False],
        ep=[1000],
        per=[1],
        max_episodes=[1000],
    )
    config_list.extend(grid_to_list(env_grid))

    return config_list
