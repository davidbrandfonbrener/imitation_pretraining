"""Replay buffer using jax sampling."""
import datetime
from typing import Dict, Optional, List
from pathlib import Path
import shutil
from collections import deque
import json

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax
import dm_env

from imitation_pretraining.data_utils.preprocess_data import (
    process_episode_tf,
    encode_observations,
)
from imitation_pretraining.data_utils import Batch

# Prevent tf from using GPU since we are training with JAX.
tf.config.set_visible_devices([], "GPU")


def get_most_recent_episodes(replay_dir: Path, max_episodes: int) -> List:
    """Get the most recent episodes."""
    ep_fns = sorted(replay_dir.glob("*.tfrecord"), reverse=True)
    return [str(ep_fn) for ep_fn in ep_fns[:max_episodes]]


class ReplayBufferStorage:
    """Object that stores episodes in a directory."""

    def __init__(
        self,
        data_specs: Dict,
        replay_dir: Path,
        clean_dir: bool = False,
        write_json: bool = True,
    ):
        """
        Args:
            data_specs: A dict of dm_env specs or dicts of specs.
            replay_dir: A pathlib.Path object.
            clean_dir: If True, delete all files in replay_dir.
            write_json: If True, write a json file with the data specs.
        """
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        if clean_dir:
            shutil.rmtree(replay_dir, ignore_errors=True)
        replay_dir.mkdir(exist_ok=True, parents=True)
        self._current_episode = []
        self._preload()

        def sample_tensor(spec):
            arr = np.zeros((2, *spec.shape), dtype=spec.dtype)
            return tf.convert_to_tensor(arr)

        def features_from_array(arr):
            return tfds.features.Tensor(
                shape=(None, *arr.shape[1:]),
                dtype=arr.dtype,
                encoding=tfds.features.Encoding.ZLIB,
            )

        example = jax.tree_map(sample_tensor, data_specs)
        feat_spec = jax.tree_map(features_from_array, example)
        self.features = tfds.features.FeaturesDict(feat_spec)

        if write_json:
            feat_json = self.features.to_json()
            json_object = json.dumps(feat_json, indent=4)
            with open(replay_dir / "features.json", "w", encoding="utf-8") as outfile:
                outfile.write(json_object)

    def __len__(self) -> int:
        return self._num_transitions

    def add(self, timestep: dm_env.TimeStep):
        """Add a timestep to the current episode in the buffer.

        Args:
            timestep: A dm_env.TimeStep object, assumed to be sent in order within an episode.
        """
        timestep_dict = timestep._asdict()

        # Expand scalars.
        def expand_scalar(value, spec):
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            return value

        timestep_dict = jax.tree_util.tree_map(
            expand_scalar, timestep_dict, self._data_specs
        )

        # Check that the timestep is valid.
        def check_shape_and_type(value, spec):
            assert spec.shape == value.shape, (spec.shape, value.shape)
            assert spec.dtype == value.dtype, (spec.dtype, value.dtype)

        jax.tree_util.tree_map(check_shape_and_type, timestep_dict, self._data_specs)

        # Add timestep to episode.
        self._current_episode.append(timestep_dict)

        # Add episode to buffer if it is done.
        if timestep.last():
            # Stack list of pytrees into a pytree of batched arrays.
            episode = jax.tree_util.tree_map(
                lambda *arrays: np.stack(arrays),
                self._current_episode[0],
                *self._current_episode[1:],
            )
            self._current_episode = []
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for filename in self._replay_dir.glob("*.tfrecord"):
            # for filename in self._replay_dir.glob("*.tf_data"):
            _, _, eps_len = filename.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode: Dict):
        eps_idx = self._num_episodes
        eps_len = episode["action"].shape[0] - 1  # subtract 1 for dummy transition
        self._num_episodes += 1
        self._num_transitions += eps_len
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{timestamp}_{eps_idx}_{eps_len}"
        # save_episode_tf(episode, str(self._replay_dir / f"{eps_fn}.tfdata"))

        eps_fn = str(self._replay_dir / f"{eps_fn}.tfrecord")
        with tf.io.TFRecordWriter(eps_fn) as writer:
            ex_bytes = self.features.serialize_example(episode)
            writer.write(ex_bytes)


class ReplayBufferDataset:
    """Object that loads episodes from a directory into a dataset."""

    def __init__(
        self,
        replay_dir: Path,
        batch_size: int,
        max_episodes: int,
        shuffle_and_repeat: bool = True,
        nstep: int = 1,
        discount: np.float32 = 1.0,
        seed: int = 0,
        cache: bool = False,
        online: bool = False,
        fetch_every: Optional[int] = None,
        delete_data: bool = False,
        jax_device_queue: bool = False,
        history: int = 1,
        average_actions: Optional[int] = None,
        encoder=None,
    ):
        """
        Args:
            replay_dir: A pathlib.Path object, where episodes are stored.
            batch_size: The number of transitions to return in a batch.
            max_episodes: The maximum number of episodes to load into memory.
            shuffle_and_repeat: If True, shuffle and repeat the dataset.
            nstep: The number of steps to use for n-step returns.
            discount: The discount factor.
            seed: A random seed.
            cache: If not None, the path to cache the dataset.
            online: If True, load episodes from replay_dir as they are generated,
                otherwise only load once.
            fetch_every: The frequency that filenames are re-read from the replay_dir.
                Only used if online=True. In units of batches.
            delete_data: If True, delete files in replay_dir once they are loaded.
            jax_device_queue: If True, use a jax device queue to prefetch data.
            history: the length of history to use for frame stacking.
            average_actions: If True, average actions over the history.
            encoder: a callable that takes in a batch of observations and returns
                a batch of encoded observations.
        """
        assert isinstance(replay_dir, Path)
        self._replay_dir = replay_dir
        self._size = 0
        self._max_episodes = max_episodes
        self._shuffle_and_repeat = shuffle_and_repeat
        self._cache = cache
        self._seed = seed
        self._nstep = nstep
        self._discount = discount
        self._batch_size = batch_size
        self._online = online
        self._fetch_every = fetch_every
        self._batches_since_fetch = 0
        self._delete_data = delete_data
        self._jax_device_queue = jax_device_queue
        self._history = history
        self._average_actions = average_actions
        self._encoder = encoder

        # Load data spec from disk.
        filename = list(self._replay_dir.glob("*.json"))[0]
        with open(filename, "r", encoding="utf-8") as read_file:
            feature_json = json.load(read_file)
        self.features = tfds.features.FeatureConnector.from_json(feature_json)

        # Create dataset.
        dataset = self._load_dataset()
        if self._encoder is not None:
            dataset = self._encode_dataset(dataset)
        self.dataset = self._shuffle_batch_and_prefetch(dataset, self._batch_size)

    def __len__(self):
        return self._size

    def _ep_to_transitions(self, episode: Dict) -> Dict:
        processed_episode = process_episode_tf(
            episode,
            self._discount,
            self._nstep,
            self._history,
            average_actions=self._average_actions,
        )
        episode_dataset = tf.data.Dataset.from_tensor_slices(processed_episode)
        return episode_dataset

    def _load_dataset(self) -> tf.data.Dataset:
        # Load episode filenames
        eps_fns = get_most_recent_episodes(self._replay_dir, self._max_episodes)
        fn_dataset = tf.data.Dataset.from_tensor_slices(eps_fns)
        if self._shuffle_and_repeat:  # Shuffle filenames
            fn_dataset = fn_dataset.shuffle(len(eps_fns), seed=self._seed)
        # Read and deserialize episodes
        dataset = tf.data.TFRecordDataset(fn_dataset, num_parallel_reads=4)
        dataset = dataset.map(
            self.features.deserialize_example, num_parallel_calls=tf.data.AUTOTUNE
        )
        # Cache episodes in RAM
        if self._cache:
            dataset = dataset.cache()
        # Process episodes into transitions
        dataset = dataset.interleave(
            self._ep_to_transitions, num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset

    def _shuffle_batch_and_prefetch(
        self, dataset: tf.data.Dataset, batch_size: int
    ) -> tf.data.Dataset:
        if self._shuffle_and_repeat:
            dataset = dataset.shuffle(10000, seed=self._seed)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset.as_numpy_iterator()

    def _encode_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.batch(100).as_numpy_iterator()
        encoded_dataset = None
        for batch in dataset:
            encoded_batch = encode_observations(batch, self._encoder)
            batch_dataset = tf.data.Dataset.from_tensor_slices(encoded_batch)
            if encoded_dataset is None:
                encoded_dataset = batch_dataset
            else:
                encoded_dataset = encoded_dataset.concatenate(batch_dataset)
        encoded_dataset = encoded_dataset.cache()
        return encoded_dataset

    def _load_episode(self, eps_fn: tf.Tensor) -> tf.data.Dataset:
        data = tf.data.TFRecordDataset(eps_fn)
        data = data.map(self.features.deserialize_example)
        episode = next(iter(data))
        return episode

    def __iter__(self) -> Batch:
        if self._jax_device_queue:
            # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
            device_queue = deque()
            device = jax.devices()[0]

            def enqueue(n):
                for _ in range(n):
                    batch = next(self.dataset)
                    device_batch = jax.device_put(batch, device)
                    device_queue.append(device_batch)

            enqueue(2)  # Queue of 2 batches is sufficient according to flax docs.
            while device_queue:
                yield device_queue.popleft()
                enqueue(1)
                if self._online:
                    self._batches_since_fetch += 1
                    if self._batches_since_fetch > self._fetch_every:
                        self._batches_since_fetch = 0
                        print("Fetching new episodes...")
                        self.dataset = self._load_dataset()
        else:
            while True:
                yield next(self.dataset)
                if self._online:
                    self._batches_since_fetch += 1
                    if self._batches_since_fetch > self._fetch_every:
                        self._batches_since_fetch = 0
                        print("Fetching new episodes...")
                        self.dataset = self._load_dataset()
