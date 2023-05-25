"""Utilities for running experiments."""
import os
import time
from typing import Dict, Optional
import csv
import pickle
import wandb


def name_experiment(config: Dict) -> str:
    """Name experiment based on config."""
    name = config["agent_name"]
    name += f"_{config['observation_adapter_name']}"
    return name


def make_path(path: str, config: Dict) -> str:
    """Create a directory inside of path using a name derived from config."""
    full_path = os.path.join(
        path, time.strftime("%b-%Y"), str(config["sweep_id"]), str(config["job_id"])
    )
    os.makedirs(full_path, exist_ok=True)
    return full_path


def most_recent_subdir(path: str) -> str:
    """Get the most recent subdirectory of path."""
    subdirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    return max(subdirs, key=os.path.getmtime)


def get_path_from_config(
    path: str,
    config: Dict,
    file_day: Optional[str] = None,
    file_time: Optional[str] = None,
) -> str:
    """Get the path from the name of the experiment.

    Returns most recent path if day and time are not specified.
    """
    if file_day is None:
        path = most_recent_subdir(path)
    else:
        path = os.path.join(path, file_day)
    path = os.path.join(path, name_experiment(config), str(config["seed"]))
    if file_time is None:
        path = most_recent_subdir(path)
    else:
        path = os.path.join(path, file_time)
    return path


class Logger:
    """Logger that writes to a csv file and wandb."""

    def __init__(self, config, csv_path, group=None, use_wandb=False):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init()  # set to your wandb account
        with open(os.path.join(csv_path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

        self.tables = ["train", "val", "eval"]
        self.csv_files = {}
        self.csv_init = {}
        for t in self.tables:
            self.csv_files.update(
                {
                    t: open(
                        os.path.join(csv_path, f"{t}.csv"),
                        "w",
                        newline="",
                        encoding="utf8",
                    )
                }
            )
            self.csv_init.update({t: False})
        self.csv_writers = {}

    def write(self, info_dict: Dict, table: str, step: int):
        """Write to csv and wandb.

        Args:
            info_dict (Dict): a dictionary of information to log.
            table (str): the table to write to.
            step (int): the index of the step.
        """
        if self.use_wandb:
            # write to wandb
            wandb_dict = {}
            for k, v in info_dict.items():
                wandb_dict.update({f"{table}/{k}": v})
            wandb.log(wandb_dict)

        # write to csv
        assert table in self.tables
        dict.update({"step": step})
        if not self.csv_init[table]:
            fieldnames = info_dict.keys()
            self.csv_writers.update(
                {table: csv.DictWriter(self.csv_files[table], fieldnames=fieldnames)}
            )
            self.csv_writers[table].writeheader()
            self.csv_init[table] = True

        self.csv_writers[table].writerow(info_dict)
        self.csv_files[table].flush()

    def close(self):
        """Close the csv files."""
        for t in self.tables:
            self.csv_files[t].close()
        wandb.finish()
