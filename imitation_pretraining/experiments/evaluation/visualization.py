"""Utility functions for visualizing the results of the evaluation."""
import os
import numpy as np
import functools as ft

from imitation_pretraining.experiments.evaluation import loading


def load_csv(log_path, idx, mode="eval"):
    csv = f"{mode}.csv"
    return np.genfromtxt(
        os.path.join(log_path, str(idx), csv), names=True, delimiter=","
    )


def get_returns(
    root_path,
    date,
    sweep_id,
    filter_func,
    label_func,
    csv_key="eval_success",
    mode="eval",
    avg=False,
):
    configs = loading.load_sweep_configs(root_path, date, sweep_id)
    log_path = os.path.join(root_path, "logs", date, sweep_id)
    returns = []

    for c in configs:
        idx = c["job_id"]
        if filter_func(c):
            env = c["eval_env_name"]
            enc = c["encoder_name"]
            try:
                eval_csv = load_csv(log_path, idx, mode=mode)
                if avg:
                    ret = np.mean(eval_csv[csv_key][-10:])
                else:
                    if len(eval_csv[csv_key].shape) > 0:
                        ret = eval_csv[csv_key][-1]
                    else:
                        ret = eval_csv[csv_key]
                returns.append((c["seed"], label_func(c), ret))
                if np.isnan(ret):
                    print(idx, env, enc)
            except:
                print("Failed: ", idx, env, enc)
                returns.append((c["seed"], label_func(c), np.nan))

    keys = sorted(list(set(r[1] for r in returns)))
    results = {key: [] for key in keys}
    for r in returns:
        results[r[1]].append(r[2])

    means, stderrs = [], []
    for k, v in results.items():
        means.append(np.nanmean(v))
        stderrs.append(np.nanstd(v) / np.sqrt(len(v)))
    return np.array(means), np.array(stderrs), np.array(keys)


def get_avg(full_results, norms=None):
    avg_results = {}
    algs = list(list(full_results.values())[0].keys())
    if norms is None:
        norms = {env: 1 for env in full_results}
    for alg in algs:
        avg_results[alg] = {}
        m = np.vstack(
            [full_results[env][alg]["means"] / norms[env] for env in full_results]
        )
        avg_results[alg]["means"] = np.mean(m, axis=0)
        s = np.vstack(
            [full_results[env][alg]["stds"] / norms[env] for env in full_results]
        )
        avg_results[alg]["stds"] = np.mean(s, axis=0)
    return avg_results


def get_full_results(
    root_path, envs, filter_function, label_function, args_dict, avg=True
):
    full_results = {}
    for i, env_name in enumerate(envs):
        env_results = {}
        for alg_key, alg_config in args_dict.items():
            for load_id in alg_config["sweep_ids"]:
                means, stderrs, keys, = get_returns(
                    root_path,
                    alg_config["date"],
                    load_id,
                    ft.partial(
                        filter_function, env_name=env_name, **alg_config["filter_args"]
                    ),
                    label_function,
                    csv_key=alg_config.get("csv_key", "eval_success"),
                    mode=alg_config.get("csv_mode", "eval"),
                )
                if len(means) > 0:
                    env_results[alg_key] = {
                        "keys": keys,
                        "means": means,
                        "stds": stderrs,
                    }

        full_results.update({env_name: env_results})

    if avg:
        return full_results, get_avg(full_results)
    else:
        return full_results


def get_args_dict(date, sweep_ids, pop_unlearned=False):
    args_dict = {
        "States": {"filter_args": {"mode": "scratch", "obs_name": "position"}},
        "Pixels + Aug": {"filter_args": {"mode": "scratch", "obs_name": "pixels_crop"}},
        "timm": {"filter_args": {"mode": "finetune", "enc_name": "timm"}},
        "r3m": {"filter_args": {"mode": "finetune", "enc_name": "r3m"}},
        "inverse_dynamics": {
            "filter_args": {
                "mode": "finetune",
                "enc_name": "inverse_dynamics",
                "nstep": 1,
            }
        },
        "bc": {"filter_args": {"mode": "finetune", "enc_name": "bc", "nstep": 1}},
        "reconstruction": {
            "filter_args": {
                "mode": "finetune",
                "enc_name": "reconstruction",
                "nstep": 1,
            }
        },
        "contrastive": {
            "filter_args": {"mode": "finetune", "enc_name": "contrastive", "nstep": 1}
        },
        "SimCLR": {
            "filter_args": {"mode": "finetune", "enc_name": "contrastive", "nstep": 0}
        },
    }

    for k in args_dict:
        args_dict[k]["date"] = date
        args_dict[k]["sweep_ids"] = sweep_ids

    if pop_unlearned:
        args_dict.pop("States")
        args_dict.pop("Pixels + Aug")
        args_dict.pop("timm")
        args_dict.pop("r3m")
    return args_dict
