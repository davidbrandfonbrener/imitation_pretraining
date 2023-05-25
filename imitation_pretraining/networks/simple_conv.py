"""A simple conv net encoder.
Based on:
https://github.com/ikostrikov/jaxrl2/blob/main/jaxrl2/networks/encoders/d4pg_encoder.py
"""
from typing import Sequence, Callable

import flax.linen as nn
import jax.numpy as jnp

# pylint: disable=arguments-differ


FRAME_TO_EMBED_DIM = {84: 16, 120: 25}


class SimpleConv(nn.Module):
    """Simple conv net, inspired by D4PG."""

    features: Sequence[int] = (32, 64, 128, 256)
    filters: Sequence[int] = (3, 3, 3, 3)
    strides: Sequence[int] = (2, 2, 1, 1)
    padding: str = "VALID"
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)
        assert len(x.shape) == 4
        del train

        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                padding=self.padding,
            )(x)
            x = self.activation_fn(x)
        return x


class SimpleDeconv(nn.Module):
    """Simple deconv net."""

    frame_size: int = 84
    dense_channels: int = 2
    features: Sequence[int] = (256, 128, 64, 32)
    filters: Sequence[int] = (3, 3, 3, 3)
    strides: Sequence[int] = (1, 1, 2, 2)
    padding: str = "VALID"
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)
        assert len(x.shape) == 2
        assert self.frame_size in [84, 120]
        del train

        # Reshape to image tensor.
        embed_dim = FRAME_TO_EMBED_DIM[self.frame_size]
        x = nn.Dense(embed_dim * embed_dim * self.dense_channels)(x)
        x = self.activation_fn(x)
        x = jnp.reshape(x, (-1, embed_dim, embed_dim, self.dense_channels))

        # Deconvolutional layers.
        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            x = nn.ConvTranspose(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                padding=self.padding,
            )(x)
            x = self.activation_fn(x)

        # Final layer to output 3 channel image.
        x = nn.ConvTranspose(
            3,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding=self.padding,
        )(x)

        assert x.shape[1] == self.frame_size
        return x
