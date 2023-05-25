"""Utils for running sweeps using slurm."""
import itertools
import os
import pickle
import time


ROOT_DIR = "PATH TO LOGS + CKPTS"


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    return list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )


def grid_to_list_of_grids(grid):
    """Convert a grid to a list of configs."""
    grid_list = grid_to_list(grid)
    return [{k: [v] for k, v in grid.items()} for grid in grid_list]


def log_gridlist_and_config(grid_list, config):
    """Log sweep grid."""
    log_path = os.path.join(
        config["project_dir"],
        "logs",
        time.strftime("%b-%Y"),
        str(config["sweep_id"]),
    )
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "sweep_grid.pkl"), "wb") as f:
        pickle.dump(grid_list, f)
    with open(os.path.join(log_path, "config.pkl"), "wb") as f:
        pickle.dump(config, f)


def write_slurm_file(n_jobs, slurm_name, mode="GPU", hrs=3):
    """Write a SLURM file for running a grid sweep."""
    assert mode in ["GPU", "CPU"]

    work_dir = os.getcwd()
    with open(
        os.path.join(work_dir, "slurm", f"{slurm_name}.slurm"), "w", encoding="utf-8"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={slurm_name}\n")
        f.write("#SBATCH --open-mode=append\n")
        if hrs >= 10:
            f.write(f"#SBATCH --time={hrs}:00:00\n")
        else:
            f.write(f"#SBATCH --time=0{hrs}:00:00\n")
        if mode == "GPU":
            f.write("#SBATCH --mem=64G\n")
            f.write("#SBATCH -c 8\n")
            f.write("#SBATCH --gres=gpu:1\n")
        else:
            f.write("#SBATCH --mem=16G\n")
            f.write("#SBATCH -c 2\n")

        if mode == "GPU":
            overlay_command = "singularity exec --nv --bind /share/apps --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash"
            gl_command = "export MUJOCO_GL=egl"
        elif mode == "CPU":
            overlay_command = "singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash"
            gl_command = "export MUJOCO_GL=osmesa"

        f.write(f"#SBATCH --array=1-{n_jobs}\n")

        f.write(
            f'{overlay_command} -c "\n source /ext3/env.sh \n {gl_command} \n conda activate imitation_pretraining \n python $HOME/imitation_pretraining/sweeps/{slurm_name} $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID \n" '
        )


def update_data_dirs(config: dict) -> dict:
    """Make an updated config after indexing into the grid."""
    env_name = config["eval_env_name"]
    pretrain_name = "pretrain" if config["pretrain"] else "finetune"

    if env_name.startswith("kitchen"):
        if env_name.split("_")[-1] == "state":
            env_name = "_".join(env_name.split("_")[:-1])

        if config["pretrain"]:  # Only pretrain from common pretraining dataset
            env_name = "kitchen_split_0"

        if config.get("kitchen_ntasks", None) is not None:
            filename = f"ep-{config['ep']}-seed-{config['seed']}-ntasks-{config['kitchen_ntasks']}"
        else:
            filename = f"ep-{config['ep']}-seed-{config['seed']}"

        if config.get("all_data", False):
            filename += "-all_data"

        config["train_data_dir"] = (
            config["project_dir"] / f"data/{env_name}/{pretrain_name}/" / filename
        )
        config["eval_data_dir"] = (
            config["project_dir"]
            / f"data/{env_name}/{pretrain_name}/"
            / f"{filename}-eval"
        )
    else:
        config["train_data_dir"] = (
            config["project_dir"]
            / f"data/{env_name}/{pretrain_name}/"
            / f"ep-{config['ep']}-per-{config['per']}-seed-{config['seed']}-expert"
        )
        config["eval_data_dir"] = (
            config["project_dir"]
            / f"data/{env_name}/{pretrain_name}/"
            / f"ep-{config['ep']}-per-{config['per']}-seed-{config['seed']}-expert-eval"
        )

    return config


def find_encoder_matches(sweep_config, encoder_configs):
    """Find encoder configs that match the sweep config."""
    matches = []
    for encoder_config in encoder_configs:
        enc_env_name = encoder_config["eval_env_name"]
        sweep_env_name = sweep_config["eval_env_name"]
        if (
            (sweep_env_name == "point_mass" and enc_env_name == "point_mass")
            or (
                sweep_env_name == "metaworld_pick_place_nogoal"
                and enc_env_name == "metaworld_pick_place_nogoal"
            )
            or ("kitchen" in sweep_env_name and enc_env_name == "kitchen_split_0")
            or (
                "door" in sweep_env_name
                and (
                    enc_env_name == "metaworld_pretrain_split_door"
                    or enc_env_name == "metaworld_pretrain_split_door-all"
                )
            )
            or (
                "metaworld" in sweep_env_name
                and "split_0" in sweep_env_name
                and (
                    "metaworld_pretrain_split_0" in enc_env_name
                    or enc_env_name == "metaworld_pretrain_split_all"
                )
            )
            or (
                "split_r3m" in sweep_env_name
                and (
                    "metaworld_pretrain_split_r3m" in enc_env_name
                    or enc_env_name == "metaworld_pretrain_split_all"
                )
            )
        ):
            matches.append(encoder_config)
    return matches
