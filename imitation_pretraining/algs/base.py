"""Base learner class to define the interface for all learners."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, Tuple, Any
import functools as ft
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import optax

from imitation_pretraining.data_utils import Batch
from imitation_pretraining.networks import PRNGKey
from imitation_pretraining.data_utils import observation_adapters
from imitation_pretraining import networks

# pylint: disable=abstract-method,not-callable

TrainState = train_state.TrainState


class BaseLearner(ABC):
    """Base learner."""

    def __init__(self, config: Dict, batch: Batch, fixed_encoder=None) -> None:
        self.config = config
        self.rng = jax.random.PRNGKey(config["seed"])
        adapter = observation_adapters.registry.make(config["observation_adapter_name"])
        self.observation_adapter = adapter
        self.action_shape = (batch.action.shape[1],)
        self.fixed_encoder = fixed_encoder
        self.state = None
        self.single_aug = config.get("single_aug", False)  # Default to separate augs

    @abstractmethod
    def update(self, batch: Batch) -> Dict:
        """Update the learner.

        Args:
            batch (Batch): a batch of input data.

        Returns:
            Dict: an information dictionary for logging purposes.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, batch: Batch) -> Dict:
        """Evaluate the learner.

        Args:
            batch (Batch): a batch of input data.

        Returns:
            Dict: an information dictionary for logging purposes.
        """
        raise NotImplementedError

    @property
    def encoder_params(self):
        """Return encoder parameters for fine-tuning."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, observation: Dict) -> jnp.ndarray:
        """Encode observation using the feature extractor.

        Args:
            observation (Dict): a dictionary of one input observation.

        Returns:
            jnp.ndarray: encoded observation.
        """
        raise NotImplementedError

    def sample_action(self, observation: Dict) -> jnp.ndarray:
        """Sample action from the policy for rollouts.

        Args:
            observation (Dict): a dictionary of one input observation.
                Each key is a string and each value is a jnp.ndarray
                of shape [*obs_shape].

        Returns:
            jnp.ndarray [*action_shape]: sampled actions.
        """
        del observation
        return np.zeros(self.action_shape)  # dummy actions by default

    def init_model(
        self,
        network_name: str,
        model_input: Any,
        init_args: Dict,
    ) -> Tuple[TrainState, Callable]:
        """Initialize model.

        Args:
            network_name (str): name of the network.
            model_input (Any): input to the network.
            init_args (Dict): a dictionary of arguments for the network
                initialization.

        Returns:
            TrainState: a flax TrainState object.
        """
        self.rng, key = jax.random.split(self.rng)
        model, variables = networks.registry.make(
            network_name, key, model_input, init_args=init_args
        )
        lr_schedule = optax.cosine_decay_schedule(
            self.config["lr"] * self.config["batch_size"] / 256,
            self.config["num_steps"],
        )
        tx = optax.adamw(learning_rate=lr_schedule)
        state = TrainState.create(
            apply_fn=model.apply, params=variables["params"], tx=tx
        )
        return (state, model)

    def adapt(self, batch: Batch, train: bool) -> Batch:
        """Adapt the batch of observations.

        Args:
            batch (Batch): a batch of input data.
            train (bool): whether to adapt the batch for training.

        Returns:
            Batch: an adapted batch of input data.
        """
        if self.single_aug:
            # Use same key for obs and next_obs
            key, self.rng = jax.random.split(self.rng)
            adapted_obs, _ = self.observation_adapter(
                key, batch.observation, train=train
            )
            adapted_next_obs, _ = self.observation_adapter(
                key, batch.next_observation, train=train
            )
        else:
            # Use different keys across obs and next_obs (default)
            adapted_obs, self.rng = self.observation_adapter(
                self.rng, batch.observation, train=train
            )
            adapted_next_obs, self.rng = self.observation_adapter(
                self.rng, batch.next_observation, train=train
            )
        batch = batch._replace(
            observation=adapted_obs, next_observation=adapted_next_obs
        )
        return batch

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save the learner.

        Args:
            path (str): path to save the learner.
            step (int): current training step.
        """
        ckpt_path = checkpoints.save_checkpoint(
            os.path.join(path, "state"), self.state, step=step, overwrite=True
        )
        print("Saved checkpoint to ", ckpt_path)

    def restore_checkpoint(self, path: str) -> None:
        """Load the learner.

        Args:
            path (str): path to load the learner.
        """
        self.state = checkpoints.restore_checkpoint(
            os.path.join(path, "state"), self.state
        )
        print("Restored checkpoint from ", path)


def apply(
    state: TrainState,
    observation: Dict,
    rng: PRNGKey,
    method: Optional[Callable] = None,
) -> jnp.ndarray:
    """Apply method to batch of observations using params from state.
    Defaults to the apply_fn method of state."""
    rng, key = jax.random.split(rng)
    output = state.apply_fn(
        {"params": state.params},
        observation,
        train=False,
        rngs={"dropout": key},
        method=method,
    )
    return output, rng


apply_jit = jax.jit(apply, static_argnames=("method",))


def grad_norm(grads) -> float:
    """Compute the norm of gradients."""
    tree_norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    return jnp.sum(jnp.array(jax.tree_util.tree_leaves(tree_norms)))
