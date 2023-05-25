"""MSE behavior cloning for continuous control."""
import os
from typing import Dict, Tuple, Callable, Optional
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core.frozen_dict import freeze
from flax import traverse_util
import optax

from imitation_pretraining.algs.base import (
    BaseLearner,
    TrainState,
    apply_jit,
    grad_norm,
)
from imitation_pretraining.data_utils import Batch
from imitation_pretraining.networks import PRNGKey, Params

# pylint: disable=not-callable, abstract-method


class JointTrainState(struct.PyTreeNode):
    """Joint training state for behavior cloning model."""

    policy: TrainState
    target_predictor: Optional[TrainState] = None  # for debugging purposes


class StaticState(struct.PyTreeNode):
    """Static state for behavior cloning model."""

    encode_fn: Callable
    predict_target: bool = False


def evaluate_loss(
    state: JointTrainState,
    batch: Batch,
    rng: PRNGKey,
    static_state: StaticState,
    train=False,
) -> Tuple[Params, Dict, PRNGKey]:
    """Evaluate loss."""
    rng, key = jax.random.split(rng)

    def loss_fn(
        policy_params: Params, target_predictor_params: Params
    ) -> Tuple[jnp.ndarray, Dict]:
        actions = state.policy.apply_fn(
            {"params": policy_params},
            batch.observation,
            train=train,
            rngs={"dropout": key},
        )
        if len(batch.action.shape) == 3:
            target_action = batch.action[:, :, 0]
        else:
            target_action = batch.action
        actor_loss = ((actions - target_action) ** 2).mean()

        if static_state.predict_target:
            embed = state.policy.apply_fn(
                {"params": policy_params},
                batch.observation,
                train=train,
                rngs={"dropout": key},
                method=static_state.encode_fn,
            )
            embed = jax.lax.stop_gradient(embed)  # Don't backprop through the encoder
            target_preds = state.target_predictor.apply_fn(
                {"params": target_predictor_params},
                embed,
                train=train,
                rngs={"dropout": key},
            )
            target_loss = ((target_preds - batch.observation["target"]) ** 2).mean()
        else:
            target_loss = 0.0

        loss = actor_loss + target_loss
        return loss, {
            "loss": loss,
            "actor_loss": actor_loss,
            "target_loss": target_loss,
        }

    argnums = (0, 1)
    if static_state.predict_target:
        target_predictor_params = state.target_predictor.params
    else:  # Don't differentiate wrt target predictor
        argnums = argnums[:-1]
        target_predictor_params = None
    grad_fn = jax.grad(loss_fn, argnums=argnums, has_aux=True)
    grads, info = grad_fn(state.policy.params, target_predictor_params)
    info = dict(info, grad_norm=grad_norm(grads))
    return (grads, info, rng)


def update(
    state: JointTrainState, batch: Batch, rng: PRNGKey, static_state: StaticState
) -> Tuple[TrainState, Dict, PRNGKey]:
    """Update parameters."""
    grads, info, rng = evaluate_loss(state, batch, rng, static_state, train=True)
    new_policy = state.policy.apply_gradients(grads=grads[0])
    if static_state.predict_target:
        new_target_predictor = state.target_predictor.apply_gradients(grads=grads[1])
    else:
        new_target_predictor = state.target_predictor
    new_state = state.replace(policy=new_policy, target_predictor=new_target_predictor)
    return (new_state, info, rng)


_update_jit = jax.jit(update, static_argnames=("static_state",))
_evaluate_loss_jit = jax.jit(evaluate_loss, static_argnames=("static_state",))


class BCLearner(BaseLearner):
    """Behavior cloning learner."""

    def __init__(self, config: Dict, batch: Batch, fixed_encoder=None):
        super().__init__(config, batch, fixed_encoder)
        observation, self.rng = self.observation_adapter(
            self.rng, batch.observation, train=False
        )

        init_args = {"action_shape": self.action_shape}
        if "multihead" in config["policy_network_name"]:
            init_args["n_heads"] = config["n_tasks"]
        policy, model = self.init_model(
            config["policy_network_name"], observation, init_args=init_args
        )
        if config["predict_target"]:
            embed = policy.apply_fn(
                {"params": policy.params}, observation, method=model.encode
            )
            target_predictor, _ = self.init_model(
                config["target_predictor_network_name"],
                embed,
                init_args={"output_dim": observation["target"].shape[-1]},
            )
        else:
            target_predictor = None
        self.state = JointTrainState(policy=policy, target_predictor=target_predictor)
        self.static_state = StaticState(
            encode_fn=model.encode, predict_target=config["predict_target"]
        )

        # Load checkpoint if path is specified
        if config["checkpoint_path"] is not None:
            assert os.path.exists(config["checkpoint_path"]), "Checkpoint not found"
            self.restore_checkpoint(config["checkpoint_path"])

        # Overwrite only the encoder with fixed_encoder params if specified
        # see https://flax.readthedocs.io/en/latest/guides/transfer_learning.html
        self.finetune_from_encoder = config["finetune_from_encoder"]
        if self.finetune_from_encoder:
            encoder_params = self.fixed_encoder.encoder_params
            params = self.state.policy.params.unfreeze()
            params["encoder"] = encoder_params
            params = freeze(params)
            self.state = self.state.replace(
                policy=self.state.policy.replace(params=params)
            )

        # Freeze encoder if desired
        if config.get("freeze_encoder", False):
            self.freeze_encoder()

        self.observation_buffer = deque(maxlen=config["history"])
        self.context = None

    def update(self, batch: Batch) -> Dict:
        batch = self.adapt(batch, train=True)
        self.state, info, self.rng = _update_jit(
            self.state, batch, self.rng, self.static_state
        )
        return jax.tree_util.tree_map(np.asarray, info)

    def eval(self, batch: Batch) -> Dict:
        batch = self.adapt(batch, train=False)
        _, info, self.rng = _evaluate_loss_jit(
            self.state, batch, self.rng, self.static_state
        )
        return jax.tree_util.tree_map(np.asarray, info)

    def sample_action(self, observation: Dict) -> jnp.ndarray:
        # Add context to observation
        if self.context is not None:
            observation = dict(observation, **self.context)

        # Add batch dimension.
        observation = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, 0), observation
        )
        # Add observation to buffer
        if self.observation_buffer.maxlen > 1:
            self.observation_buffer.append(observation)
            while len(self.observation_buffer) < self.observation_buffer.maxlen:
                self.observation_buffer.append(observation)
            # Frame stack along new final dimension.
            observation = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=-1), *list(self.observation_buffer)
            )
        # Encode observation if fixed encoder is specified
        if self.fixed_encoder is not None and not self.finetune_from_encoder:
            embedding = self.fixed_encoder.encode(observation)
            observation = dict(observation, embedding=embedding)
        # Adapt observation and sample action.
        adapted_obs, self.rng = self.observation_adapter(
            self.rng, observation, train=False
        )
        actions, self.rng = apply_jit(self.state.policy, adapted_obs, self.rng)
        return np.clip(np.asarray(actions)[0], -1, 1)

    @property
    def encoder_params(self):
        """Encoder parameters for finetuning."""
        return self.state.policy.params["encoder"]

    def encode(self, observation: Dict) -> jnp.ndarray:
        """Encode observation using the feature extractor."""
        adapted_obs, self.rng = self.observation_adapter(
            self.rng, observation, train=False
        )
        embedding, self.rng = apply_jit(
            self.state.policy, adapted_obs, self.rng, method=self.static_state.encode_fn
        )
        return embedding

    def freeze_encoder(self) -> None:
        """Freeze the encoder by changing the optimizer."""
        tx = self.state.policy.tx
        params = self.state.policy.params
        partition_optimizers = {"trainable": tx, "frozen": optax.set_to_zero()}
        param_partitions = freeze(
            traverse_util.path_aware_map(
                lambda path, v: "frozen" if "encoder" in path else "trainable",
                params,
            )
        )
        new_tx = optax.multi_transform(partition_optimizers, param_partitions)
        new_policy = TrainState.create(
            apply_fn=self.state.policy.apply_fn, params=params, tx=new_tx
        )
        self.state = self.state.replace(policy=new_policy)
