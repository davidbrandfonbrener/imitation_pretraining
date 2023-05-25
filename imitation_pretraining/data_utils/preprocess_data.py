"""Utilities for preprocessing episodes into batches of transitions."""
from typing import Dict, List, Optional
import tensorflow as tf
import jax
import jax.numpy as jnp

from imitation_pretraining.data_utils import Batch


def encode_observations(
    batch: Batch,
    encoder,
    pop_pixels: bool = True,
    encode_next_obs: bool = True,
) -> Batch:
    """Encode observation in batch using encoder_fn."""
    if type(encoder) == tuple:  # Hack to encode target as action
        encoder, target_encoder = encoder
        if target_encoder is not None:
            target = target_encoder.encode(batch.observation)
            batch = batch._replace(action=target)

    if encoder is not None:
        embedding = encoder.encode(batch.observation)
        obs = dict(batch.observation, embedding=embedding)
        if pop_pixels:
            obs.pop("pixels")
    else:
        obs = batch.observation

    if encode_next_obs and encoder is not None:
        next_embedding = encoder.encode(batch.next_observation)
        next_obs = dict(batch.next_observation, embedding=next_embedding)
        if pop_pixels:
            next_obs.pop("pixels")
    else:
        next_obs = batch.next_observation

    return batch._replace(observation=obs, next_observation=next_obs)


# Prevent tf from using GPU since we are training with JAX.
tf.config.set_visible_devices([], "GPU")


def stack_history_tf(obs: Dict, history: int) -> Dict:
    """Stack observations in history."""
    # Postpend with repeated first obs
    obs = {
        k: tf.concat([v, tf.repeat(v[:1], history - 1, axis=0)], 0)
        for k, v in obs.items()
    }
    # Stack history into last dim:  (time, ...) -> (time, ..., history)
    stacked_obs = {}
    for k, v in obs.items():
        stacked_obs[k] = tf.stack(
            [tf.roll(v, shift=i, axis=0) for i in range(history)][::-1], axis=-1
        )
    # Remove suffix of wrapped around values
    stacked_obs = {k: v[: -(history - 1)] for k, v in stacked_obs.items()}
    return stacked_obs


def stack_actions_tf(action: tf.Tensor, nstep: int) -> tf.Tensor:
    """Stack future nstep actions along new axis."""
    # Prepend with repeated last action
    action = tf.concat([tf.repeat(action[-1:], nstep - 1, 0), action], 0)
    # Stack history into last dim: (time, ...) -> (time, ..., nstep)
    rolled_action = [tf.roll(action, shift=i, axis=0) for i in range(0, -nstep, -1)]
    stacked_action = tf.stack(rolled_action, axis=-1)
    # Remove prefix of wrapped around values
    stacked_action = stacked_action[nstep - 1 :]
    return stacked_action


def sliding_average_tf(tensor, window_size):
    # Create a kernel of ones
    kernel = tf.ones(shape=(window_size, 1, 1), dtype=tf.float32) / window_size
    # Apply the convolution using the kernel
    averaged_tensor = tf.nn.conv1d(
        tf.transpose(tensor)[:, :, tf.newaxis], filters=kernel, stride=1, padding="SAME"
    )
    # Remove the extra dimension
    averaged_tensor = tf.transpose(tf.squeeze(averaged_tensor))
    return averaged_tensor


def process_episode_tf(
    episode: Dict,
    gamma: jnp.float32,
    nstep: int,
    history: int = 1,
    average_actions: Optional[int] = None,
    include_goal_pixels: bool = True,
) -> Batch:
    """Process an episode into a dict of tensors for training."""
    # Select observations and actions. Note that the first step contains
    # the initial obs, but dummy action/reward/discount.
    obs = {k: v[:-1] for k, v in episode["observation"].items()}
    if include_goal_pixels:
        goal_pixels = episode["observation"]["pixels"][-1:]

    action = episode["action"][1:]
    if average_actions is not None:
        action = sliding_average_tf(action, window_size=average_actions)
    if nstep > 1:  # Stack future actions.
        action = stack_actions_tf(action, nstep)

    if nstep == 0:  # Allow overload of nstep for static methods
        next_obs = obs
    else:
        next_obs = {
            k: tf.concat([v[nstep:], tf.repeat(v[-1:], nstep - 1, axis=0)], 0)
            for k, v in episode["observation"].items()
        }  # Repeat last obs nstep times at end of episode.

    # Stack observation history.
    if history > 1:
        obs = stack_history_tf(obs, history)
        next_obs = stack_history_tf(next_obs, history)

    # Add goal pixels to obs only
    if include_goal_pixels:
        obs["goal_pixels"] = tf.repeat(goal_pixels, len(obs["pixels"]), axis=0)
        next_obs["goal_pixels"] = tf.repeat(
            goal_pixels, len(next_obs["pixels"]), axis=0
        )

    # Compute n step returns
    padded_reward = tf.concat(
        [episode["reward"], tf.zeros_like(episode["reward"][:nstep])], 0
    )
    padded_discount = tf.concat(
        [episode["discount"], tf.zeros_like(episode["discount"][:nstep])], 0
    )
    reward = tf.zeros_like(episode["reward"][1:])
    discount = tf.ones_like(episode["discount"][1:])
    ep_len = len(reward)
    for i in range(1, nstep + 1):
        reward += discount * padded_reward[i : i + ep_len]
        discount *= padded_discount[i : i + ep_len] * gamma

    return Batch(obs, action, reward, discount, next_obs)
