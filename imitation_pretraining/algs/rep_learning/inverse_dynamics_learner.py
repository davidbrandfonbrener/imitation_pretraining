"""Representation learning using inverse dynamics."""
import os
from typing import Dict, Tuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from imitation_pretraining.algs.base import (
    BaseLearner,
    TrainState,
    apply_jit,
    grad_norm,
)
from imitation_pretraining.data_utils import Batch
from imitation_pretraining.networks import PRNGKey, Params

# pylint: disable=unused-argument, abstract-method, not-callable


class JointTrainState(struct.PyTreeNode):
    """Joint training state for inverse dynamics model."""

    phi: TrainState
    policy: TrainState
    target_predictor: TrainState  # for debugging purposes


class StaticState(struct.PyTreeNode):
    """Static state for inverse dynamics model."""

    stop_grad: bool
    predict_all_actions: bool = False


def evaluate_loss(
    state: JointTrainState,
    batch: Batch,
    rng: PRNGKey,
    static_state: StaticState,
    train: bool = False,
) -> Tuple[Params, Dict, PRNGKey]:
    """Evaluate loss."""
    rng, key = jax.random.split(rng)

    def loss_fn(
        phi_params: Params, policy_params: Params, target_predictor_params: Params
    ) -> Tuple[jnp.ndarray, Dict]:
        phi_embeddings = state.phi.apply_fn(
            {"params": phi_params},
            batch.observation,
            train=train,
            rngs={"dropout": key},
        )
        psi_embeddings = state.phi.apply_fn(
            {"params": phi_params},
            batch.next_observation,
            train=train,
            rngs={"dropout": key},
        )
        if static_state.stop_grad:
            psi_embeddings = jax.lax.stop_gradient(psi_embeddings)
        concat_embeddings = jnp.concatenate([phi_embeddings, psi_embeddings], axis=-1)

        action_preds = state.policy.apply_fn(
            {"params": policy_params},
            {"state": concat_embeddings},
            train=train,
            rngs={"dropout": key},
        )
        if len(batch.action.shape) == 2 or static_state.predict_all_actions:
            action_loss = jnp.mean((action_preds - batch.action) ** 2)
        elif len(batch.action.shape) == 3:  # Only predict first of stacked actions
            action_loss = jnp.mean((action_preds - batch.action[:, :, 0]) ** 2)
        else:
            raise ValueError("Invalid action shape.")

        # Debugging loss, no gradients passed to encoder
        target_preds = state.target_predictor.apply_fn(
            {"params": target_predictor_params},
            jax.lax.stop_gradient(phi_embeddings),
            train=train,
            rngs={"dropout": key},
        )
        target_loss = ((target_preds - batch.observation["target"]) ** 2).mean()

        loss = action_loss + target_loss
        return loss, {
            "loss": loss,
            "action_loss": action_loss,
            "target_loss": target_loss,
            "embedding_norm": jnp.linalg.norm(phi_embeddings[0]),
        }

    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2), has_aux=True)
    grads, info = grad_fn(
        state.phi.params, state.policy.params, state.target_predictor.params
    )
    info = dict(info, grad_norm=grad_norm(grads))
    return (grads, info, rng)


def update(
    state: JointTrainState, batch: Batch, rng: PRNGKey, static_state: StaticState
) -> Tuple[TrainState, Dict, PRNGKey]:
    """Update parameters."""
    grads, info, rng = evaluate_loss(state, batch, rng, static_state, train=True)
    new_phi = state.phi.apply_gradients(grads=grads[0])
    new_policy = state.policy.apply_gradients(grads=grads[1])
    new_target_predictor = state.target_predictor.apply_gradients(grads=grads[2])
    new_state = state.replace(
        phi=new_phi, policy=new_policy, target_predictor=new_target_predictor
    )
    return (new_state, info, rng)


_update_jit = jax.jit(update, static_argnames=("static_state",))
_evaluate_loss_jit = jax.jit(evaluate_loss, static_argnames=("static_state",))


class RepLearner(BaseLearner):
    """representation learner."""

    def __init__(self, config: Dict, batch: Batch, fixed_encoder=None):
        super().__init__(config, batch, fixed_encoder)
        observation, self.rng = self.observation_adapter(
            self.rng, batch.observation, train=False
        )

        phi, _ = self.init_model(
            config["phi_network_name"],
            observation,
            init_args={},
            # freeze_model=config.get("freeze_encoder", False),
        )
        features = phi.apply_fn({"params": phi.params}, observation)
        concat_features = jnp.concatenate([features, features], axis=-1)
        if config["predict_all_actions"]:
            self.action_shape = batch.action.shape[1:]
        policy, _ = self.init_model(
            config["policy_network_name"],
            {"state": concat_features},
            init_args={"action_shape": self.action_shape},
        )
        target_predictor, _ = self.init_model(
            config["target_predictor_network_name"],
            features,
            init_args={"output_dim": observation["target"].shape[-1]},
        )
        self.state = JointTrainState(
            phi=phi, policy=policy, target_predictor=target_predictor
        )
        self.static_state = StaticState(
            stop_grad=config["stop_grad"],
            predict_all_actions=config["predict_all_actions"],
        )

        # Load checkpoint if path is specified
        if config["checkpoint_path"] is not None:
            assert os.path.exists(config["checkpoint_path"]), "Checkpoint not found"
            self.restore_checkpoint(config["checkpoint_path"])

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

    @property
    def encoder_params(self):
        return self.state.phi.params

    def encode(self, observation: Dict) -> jnp.ndarray:
        adapted_obs, self.rng = self.observation_adapter(
            self.rng, observation, train=False
        )
        output, self.rng = apply_jit(self.state.phi, adapted_obs, self.rng)
        return output
