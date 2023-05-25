"""A simple MLP network module."""
from typing import Optional, Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp

# pylint: disable=arguments-differ


class MLP(nn.Module):
    """Simple MLP network."""

    output_dim: int
    hidden_dims: Sequence[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for size in self.hidden_dims:
            x = nn.Dense(size)(x)
            x = self.activation_fn(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.output_dim)(x)
        return x
