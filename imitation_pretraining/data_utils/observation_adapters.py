"""Registry of observation adapters."""
from typing import Dict, Sequence
import functools as ft
import jax.numpy as jnp
import jax
import chex
from imitation_pretraining.data_utils import augmentations


def adapt(
    rng: chex.PRNGKey,
    obs: Dict,
    keys: Sequence[str],
    train: bool,
    index_list: jnp.ndarray = None,  # needed for multiheaded architectures
    crop: bool = False,
    pad_size: int = 8,
    color_transform: bool = False,
    gaussian_blur: bool = False,
    add_target: bool = True,
    smooth_obs: bool = False,
) -> jnp.ndarray:
    """Adapt observation and add augmentations."""
    adapted_obs = {}

    # Add pixels.
    if "pixels" in keys:
        new_pixels = jnp.float32(obs["pixels"]) / 255.0
        assert len(new_pixels.shape) == 4, "Pixels do not have batch."
        assert new_pixels.shape[3] == 3, "Pixels do not have RGB channels."
        # Reshape to handle frame stacking.
        batch, height, width, _ = new_pixels.shape[:4]
        new_pixels = jnp.reshape(new_pixels, (batch, height, width, -1))
        # Add image augmentatons.
        if train:
            if crop:
                rng, key = jax.random.split(rng)
                new_pixels = augmentations.random_crop(new_pixels, key, pad_size)
            if color_transform:
                rng, key = jax.random.split(rng)
                new_pixels = augmentations.color_transform(
                    new_pixels,
                    key,
                    **augmentations.augment_config["view1"]["color_transform"]
                )
            if gaussian_blur:
                rng, key = jax.random.split(rng)
                new_pixels = augmentations.gaussian_blur(
                    new_pixels,
                    key,
                    **augmentations.augment_config["view1"]["gaussian_blur"]
                )
        adapted_obs["pixels"] = new_pixels

    # Add state.
    state_key_list = ["position", "velocity", "goal", "embedding", "object"]
    if any(k in keys for k in state_key_list):
        obs_list = [obs[k] for k in keys if k in state_key_list]
        if smooth_obs and len(obs_list[0].shape) == 3:  # Take mean over history
            obs_list = [jnp.mean(obs, axis=-1) for obs in obs_list]
        obs_list = [jnp.reshape(obs, (obs.shape[0], -1)) for obs in obs_list]
        adapted_obs["state"] = jnp.concatenate(obs_list, axis=-1)

    # Add task index.
    if "index" in keys:
        task_id = jnp.reshape(obs["task_id"], (obs["task_id"].shape[0], -1))[:, 0]
        if index_list is not None:
            # Map random indices to indices in range [0, num_tasks)
            # Assumes that all ids are present in index_list
            one_hot_matrix = jnp.where(task_id == index_list[:, None], 1, 0)
            index_matrix = one_hot_matrix * jnp.arange(index_list.shape[0])[:, None]
            task_id = jnp.sum(index_matrix, axis=0)
        adapted_obs["index"] = task_id

    if add_target:
        target_keys = [k for k in ["position", "object"] if k in obs]
        obs_list = [jnp.reshape(obs[k], (obs[k].shape[0], -1)) for k in target_keys]
        adapted_obs["target"] = jnp.concatenate(obs_list, axis=-1)

    if "goal_pixels" in keys:
        new_pixels = jnp.float32(obs["goal_pixels"]) / 255.0
        assert len(new_pixels.shape) == 4, "Goal pixels do not have stacking."
        # Add image augmentatons.
        if train:
            if crop:
                rng, key = jax.random.split(rng)
                new_pixels = augmentations.random_crop(new_pixels, key, pad_size)
            if color_transform:
                rng, key = jax.random.split(rng)
                new_pixels = augmentations.color_transform(
                    new_pixels,
                    key,
                    **augmentations.augment_config["view1"]["color_transform"]
                )
            if gaussian_blur:
                rng, key = jax.random.split(rng)
                new_pixels = augmentations.gaussian_blur(
                    new_pixels,
                    key,
                    **augmentations.augment_config["view1"]["gaussian_blur"]
                )
        adapted_obs["goal_pixels"] = new_pixels

    return adapted_obs, rng


jit_adapt = jax.jit(adapt, static_argnums=(2, 3, 5, 6, 7, 8, 9, 10))


class Registry(object):
    """A registry for adapters."""

    def __init__(self):
        self._adapters = {}

    def register(self, name, adapter_fn):
        """Register an adapter."""
        self._adapters[name] = adapter_fn
        return

    def make(self, name):
        """Build an adapter. Return the adapter_fn."""
        return self._adapters[name]


registry = Registry()

# Non-pixel adapters
registry.register("position", ft.partial(jit_adapt, keys=("position",)))
registry.register("position_goal", ft.partial(jit_adapt, keys=("position", "goal")))
registry.register("position_object", ft.partial(jit_adapt, keys=("position", "object")))
registry.register(
    "position_object_smooth",
    ft.partial(jit_adapt, keys=("position", "object"), smooth_obs=True),
)
registry.register(
    "position_goal_object", ft.partial(jit_adapt, keys=("position", "goal", "object"))
)
registry.register("embedding", ft.partial(jit_adapt, keys=("embedding",)))

# Pixel adapters
registry.register("pixels", ft.partial(jit_adapt, keys=("pixels")))
registry.register("pixels_crop", ft.partial(jit_adapt, keys=("pixels"), crop=True))
registry.register("pixels_goal", ft.partial(jit_adapt, keys=("pixels", "goal")))
registry.register(
    "pixels_goal_crop", ft.partial(jit_adapt, keys=("pixels", "goal"), crop=True)
)
registry.register(
    "pixels_goalpixels_crop",
    ft.partial(jit_adapt, keys=("pixels", "goal_pixels"), crop=True),
)
registry.register("pixels_position", ft.partial(jit_adapt, keys=("pixels", "position")))
registry.register("pixels_index", ft.partial(jit_adapt, keys=("pixels", "index")))
registry.register(
    "pixels_crop_index", ft.partial(jit_adapt, keys=("pixels", "index"), crop=True)
)
registry.register(
    "pixels_all_augs",
    ft.partial(
        jit_adapt, keys=("pixels"), crop=True, color_transform=True, gaussian_blur=True
    ),
)
