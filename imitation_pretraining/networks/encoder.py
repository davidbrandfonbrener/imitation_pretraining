"""Encoder modules."""
from typing import Dict, Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn

from imitation_pretraining.networks.mlp import MLP
from imitation_pretraining.networks.spatial_layers import (
    SpatialSoftmax,
    SpatialEmbedding,
)

# pylint: disable=arguments-differ,not-callable,attribute-defined-outside-init


class Encoder(nn.Module):
    """Encoder that will embed pixels, state, and (optionally) actions."""

    backbone: nn.Module
    latent_dim: int
    normalize_merge: bool = False
    spatial_softmax: Optional[int] = None
    spatial_embedding: Optional[int] = None
    action_hidden_dims: Tuple[int] = (256, 256)
    identity: bool = False

    @nn.compact
    def __call__(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Assumes obs is a dict with at least one of 'pixels' or 'state'."""
        if self.identity:  # Hack to skip encoder and return state
            state = obs["state"]
            return nn.Dense(self.latent_dim)(state)

        assert "pixels" in obs or "state" in obs
        num_inputs = int("pixels" in obs) + int("state" in obs)
        feature_dim = self.latent_dim // num_inputs
        if "pixels" in obs:
            pixels = obs["pixels"]
            embedding = self.backbone(pixels, train)

            if self.spatial_softmax is not None:
                embedding = SpatialSoftmax(self.spatial_softmax)(embedding)
                embedding = embedding.reshape((embedding.shape[0], -1))
            elif self.spatial_embedding is not None:
                embedding = SpatialEmbedding(
                    features_per_channel=self.spatial_embedding
                )(embedding)
            else:
                embedding = embedding.reshape((embedding.shape[0], -1))

            embedding = nn.Dense(feature_dim)(embedding)
            if self.normalize_merge:
                embedding = nn.LayerNorm()(embedding)
                embedding = nn.tanh(embedding)
        else:
            # Add zero-dimensional dummy embedding.
            embedding = jnp.zeros((obs["state"].shape[0], 0))
        if "state" in obs:
            state = obs["state"]
            state = nn.Dense(feature_dim)(state)
            if self.normalize_merge:
                state = nn.LayerNorm()(state)
                state = nn.tanh(state)
            embedding = jnp.concatenate([embedding, state], axis=-1)

        return embedding


class ActionEncoder(nn.Module):
    """Encoder that will embed actions."""

    latent_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout_rate: float = 0.0
    normalize_merge: bool = False

    @nn.compact
    def __call__(self, action: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Assumes action is a flattened array."""
        action = jnp.reshape(action, (action.shape[0], -1))
        action = MLP(self.latent_dim, self.hidden_dims, dropout_rate=self.dropout_rate)(
            action, train
        )
        if self.normalize_merge:
            action = nn.LayerNorm()(action)
            action = nn.tanh(action)
        return action


class Projector(nn.Module):
    """Projector for contrastive learning."""

    latent_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout_rate: float = 0.0
    normalize_projection: bool = True

    @nn.compact
    def __call__(self, embedding: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Projects embedding onto the unit sphere."""
        projection = MLP(
            self.latent_dim, self.hidden_dims, dropout_rate=self.dropout_rate
        )(embedding, train)
        if self.normalize_projection:
            norm = jnp.linalg.norm(projection, axis=-1, keepdims=True)
            projection = projection / norm
        return projection


class DoubleProjector(nn.Module):
    """Projector for contrastive learning."""

    latent_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout_rate: float = 0.0
    normalize_projection: bool = True

    @nn.compact
    def __call__(
        self, embeddings: Tuple[jnp.ndarray], train: bool = False
    ) -> jnp.ndarray:
        """Projects two embeddings onto the unit sphere."""
        assert len(embeddings) == 2
        embedding_1, embedding_2 = embeddings
        projection_1 = Projector(
            self.latent_dim,
            self.hidden_dims,
            self.dropout_rate,
            self.normalize_projection,
        )(embedding_1, train)
        projection_2 = Projector(
            self.latent_dim,
            self.hidden_dims,
            self.dropout_rate,
            self.normalize_projection,
        )(embedding_2, train)
        return projection_1, projection_2


class VAEProjector(nn.Module):
    """Projector for VAE."""

    latent_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, embedding: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Assumes obs is a dict with at least one of 'pixels' or 'state'."""
        projection = MLP(
            2 * self.latent_dim, self.hidden_dims, dropout_rate=self.dropout_rate
        )(embedding, train)
        mu, logvar = jnp.split(projection, 2, axis=-1)
        return mu, logvar
