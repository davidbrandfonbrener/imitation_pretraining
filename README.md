# Imitation Pretraining

We use python 3.9. Most dependencies can be installed by running `pip install -r requirements.txt`.

These dependencies are not installed by `pip install -r requirements.txt`:
- GPU support for jax (https://github.com/google/jax#installation)
- mujoco-py and mujoco 210 (https://github.com/openai/mujoco-py)

Our `imitation_pretraining` package can then be installed by `pip install -e .`.

Minimal examples of how to generate data, pretrain an encoder, and finetune and evaluate a policy on `point_mass` using `imitation_pretraining` are in `examples/`. These can be run as follows:
```
cd examples
python generate_data.py
python pretrain_encoder.py
python finetune_and_evaluate.py
```
This will create `data/`, `logs/`, and `ckpts/` directories inside of the `examples/` directory. Note the examples train on a very small amount of data for very few steps. This means they can run quickly, even on CPU, but will not produce good results.

The full sweeps to recreate the paper are in `sweeps/`. These scripts create slurm array jobs and were written for our internal cluster. They are provided in full for reproducibility purposes. Path variables need to be set in `sweeps/sweep_utils.py` and `sweeps/configs/config_utils.py` so that logs and checkpoints are written in an appropriate place on your machine.

Each sweep script takes two integer command line arguments: `JOB_ID` and `SWEEP_ID`. Setting `JOB_ID` to 0 will write a slurm sbatch file to launch an array job. Otherwise, the `JOB_ID` indicates the index of the job within the sweep (which have 10s to 1000s of jobs depending on the sweep) and the `SWEEP_ID` indicates the index of the entire sweep which is used to store all of the logs and checkpoints from the sweep in a directory named by the `SWEEP_ID`.

For example, we could generate one of the point mass pretraining datasets by running:
```
cd sweeps
python generate_point_mass_pretrain.py 1
```
Once we have generated *all* of the datasets, we could run a pretraining job by running:
```
cd sweeps
python pretrain_sweep_data_size.py 1 12345
```
And once a pretraining sweep has concluded, we would copy the relevant `SWEEP_ID` into the finetuning script and then could run a finetuning job by running something like:
```
cd sweeps
python bc_finetune_sweep_pt_data_size.py 1 12346
```
As stated in the paper, we estimate that reproducing all the results in the paper by creating and launching the relevant slurm sweeps took roughly 600 GPU hours on RTX8000 GPUs and about 2000 CPU hours.

The notebook used to generate the figures is also included for reproducibility purposes, but is not expected to run since the training logs are not included in the repo.

Note that all data can be generated from scratch except for the raw kitchen data which is originally from https://github.com/google-research/relay-policy-learning. We use a version of the data that was preprocessed by https://github.com/jeffacce/play-to-policy. We host an easily downloadable version of the `numpy` arrays at https://osf.io/efpvx/.  After downloading that raw data, the local path to the dataset needs to be set in `imitation_pretraining/experiments/data_generation/kitchen.py` so that the full image-based data can be generated.

Logs are always generated locally as csv, but it is also possible to simultaneously log using `wandb` by inserting your `wandb` information into the `init` call inside of the `Logger` object in `imitation_pretraining/experiments/training/training_utils.py`.


Citation:
```
TODO
```