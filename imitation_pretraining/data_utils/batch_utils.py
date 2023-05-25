"""Utilites for data processing."""
import collections
from typing import Dict
import jax.numpy as jnp


Batch = collections.namedtuple(
    "Batch", ["observation", "action", "reward", "discount", "next_observation"]
)


def obs_to_jnp(obs: Dict) -> Dict:
    """Convert an observation to JAX arrays."""
    return {k: jnp.array(v, jnp.float32) for k, v in obs.items()}


def batch_to_jnp(batch: Batch) -> Batch:
    """Convert a batch to JAX arrays."""
    return Batch(
        observation=obs_to_jnp(batch.observation),
        action=jnp.array(batch.action, jnp.float32),
        reward=jnp.array(batch.reward, jnp.float32),
        discount=jnp.array(batch.discount, jnp.float32),
        next_observation=obs_to_jnp(batch.next_observation),
    )
