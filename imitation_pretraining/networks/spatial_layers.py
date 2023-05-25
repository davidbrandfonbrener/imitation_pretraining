"""A spatial softmax and spatial embedding.
Based on:
https://rll.berkeley.edu/dsae/dsae.pdf
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/base_nets.py
https://github.com/Asap7772/PTR/blob/master/jaxrl2/networks/encoders/spatial_softmax.py
"""

from typing import Any, Callable
import flax.linen as nn
import jax.numpy as jnp

Dtype = Any
Shape = Any
Array = Any
Key = Any

# pylint: disable=arguments-differ


class SpatialSoftmax(nn.Module):
    """A spatial softmax layer."""

    n_keypoints: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert len(x.shape) == 4  # (batch, height, width, channels)
        b, h, w, _ = x.shape

        # Compute conv features for n_keypoints
        x = nn.Conv(self.n_keypoints, kernel_size=(1, 1))(x)

        # Define position grid
        pos_x, pos_y = jnp.meshgrid(jnp.linspace(-1, 1, w), jnp.linspace(-1, 1, h))

        # Reshape arrays to (b*n_keypoints, h*w)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = jnp.reshape(x, (-1, h * w))
        pos_x = jnp.reshape(pos_x, (-1, h * w))
        pos_y = jnp.reshape(pos_y, (-1, h * w))

        # Compute attention and resulting positions
        attention = nn.softmax(x, axis=-1)
        expected_x = jnp.sum(attention * pos_x, axis=-1, keepdims=True)
        expected_y = jnp.sum(attention * pos_y, axis=-1, keepdims=True)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=-1)

        # Reshape features to (b, n_keypoints, 2)
        features = jnp.reshape(expected_xy, (b, self.n_keypoints, 2))
        return features


class SpatialEmbedding(nn.Module):
    """Learned spatial embeddings layer"""

    features_per_channel: int
    kernel_init: Callable[[Key, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert len(x.shape) == 4  # (batch, height, width, channels)
        b, h, w, c = x.shape

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (w, h, c, self.features_per_channel),
            self.param_dtype,
        )
        product = jnp.expand_dims(x, -1) * jnp.expand_dims(kernel, 0)
        features = jnp.sum(product, axis=(1, 2))
        return jnp.reshape(features, [b, -1])
