"""Policy modules."""
from typing import Dict, Sequence, Optional
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from imitation_pretraining.networks import mlp

# pylint: disable=arguments-differ
# pylint: disable=attribute-defined-outside-init


class MLPPolicy(nn.Module):
    """MLP policy network for state-based observations."""

    action_shape: Sequence[int]
    hidden_dims: Sequence[int]
    dropout_rate: Optional[float] = None
    tanh_action: bool = True
    n_bins: int = 1

    def setup(self):
        """Setup."""
        self.mlp = mlp.MLP(
            np.prod(self.action_shape) * self.n_bins,
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
        )

    def encode(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Encode the observation."""
        del train, obs
        return None

    def __call__(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Forward pass."""
        x = obs["state"]
        x = self.mlp(x, train)
        if self.n_bins == 1:
            x = jnp.reshape(x, (-1,) + self.action_shape)
            if self.tanh_action:
                x = nn.tanh(x)
        else:
            x = jnp.reshape(x, (-1,) + self.action_shape + (self.n_bins,))
        return x


class EncoderPolicy(nn.Module):
    """Encoder policy network for pixel and state observations."""

    encoder: nn.Module
    action_shape: Sequence[int]
    hidden_dims: Sequence[int]
    dropout_rate: Optional[float] = None
    tanh_action: bool = True
    n_bins: int = 1

    def setup(self):
        """Setup."""
        self.mlp_policy = MLPPolicy(
            self.action_shape,
            self.hidden_dims,
            self.dropout_rate,
            self.tanh_action,
            self.n_bins,
        )

    def encode(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Encode observation."""
        return self.encoder(obs, train=train)

    def __call__(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Forward pass."""
        embedding = self.encode(obs, train)
        action = self.mlp_policy({"state": embedding}, train)
        return action


class GoalPixelsPolicy(EncoderPolicy):
    """Encoder policy network for pixel and state observations."""

    def __call__(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Forward pass with separate embedding of goal_pixels."""
        embedding = self.encode(obs, train)
        goal_embedding = self.encode({"pixels": obs["goal_pixels"]}, train)
        joint_embedding = jnp.concatenate([embedding, goal_embedding], axis=-1)
        action = self.mlp_policy({"state": joint_embedding}, train)
        return action


class MultiheadEncoderPolicy(nn.Module):
    """Encoder policy network for pixel and state observations with multiple linear heads."""

    encoder: nn.Module
    n_heads: int
    action_shape: Sequence[int]
    hidden_dims: Sequence[int]
    dropout_rate: Optional[float] = None
    tanh_action: bool = True

    def setup(self):
        """Setup."""
        self.action_dim = np.prod(self.action_shape)
        self.mlp = mlp.MLP(
            self.n_heads * self.action_dim,
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
        )

    def encode(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Encode observation."""
        return self.encoder(obs, train=train)

    def __call__(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Forward pass."""
        embedding = self.encode(obs, train)
        multihead_actions = self.mlp(embedding, train)
        multihead_actions = jnp.reshape(
            multihead_actions, (-1, self.n_heads, self.action_dim)
        )

        # Select the correct head for each sample.
        output_index = obs["index"]
        action = multihead_actions[jnp.arange(output_index.shape[0]), output_index, :]
        action = jnp.reshape(action, (-1,) + self.action_shape)
        if self.tanh_action:
            action = nn.tanh(action)
        return action


class MDNEncoderPolicy(nn.Module):
    """MDN policy network for state-based observations."""

    encoder: nn.Module
    n_comps: int
    action_shape: Sequence[int]
    hidden_dims: Sequence[int]
    dropout_rate: Optional[float] = None
    tanh_action: bool = True

    def setup(self):
        """Setup."""
        output_dim = self.n_comps * (np.prod(self.action_shape) + 1)
        self.mlp = mlp.MLP(
            output_dim,
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
        )

    def encode(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Encode observation."""
        return self.encoder(obs, train=train)

    def __call__(self, obs: Dict, train: bool = False) -> jnp.ndarray:
        """Forward pass."""
        embedding = self.encode(obs, train)
        outputs = self.mlp(embedding, train)
        mixture_logits, means = jnp.split(outputs, [self.n_comps], axis=-1)

        # Means are (batch, action_shape, n_comps)
        means = jnp.split(means, self.n_comps, axis=-1)
        means = jnp.stack(means, axis=-1)
        if self.tanh_action:
            means = nn.tanh(means)
        return mixture_logits, means
