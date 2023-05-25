"""Contrastive representation learning using dynamics."""
import os
from typing import Dict, Tuple, Optional

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
    """Joint training state for contrastive model."""

    phi: TrainState
    phi_projector: TrainState
    psi: TrainState
    psi_projector: TrainState
    policy: TrainState  # for debugging purposes
    target_predictor: TrainState  # for debugging purposes
    action_encoder: Optional[TrainState] = None


class StaticState(struct.PyTreeNode):
    """Static state for contrastive model."""

    temperature: float
    label_smoothing: float
    share_encoder: bool
    include_action: bool


def evaluate_loss(
    state: JointTrainState,
    batch: Batch,
    rng: PRNGKey,
    static_state: StaticState,
    train=False,
) -> Tuple[Params, Dict, PRNGKey]:
    """Evaluate contrastive loss."""
    rng, key = jax.random.split(rng)

    def loss_fn(
        phi_params: Params,
        phi_projector_params: Params,
        psi_params: Params,
        psi_projector_params: Params,
        policy_params: Params,  # for debugging purposes
        target_predictor_params: Params,  # for debugging purposes
        action_encoder_params: Optional[Params] = None,
    ) -> Tuple[jnp.ndarray, Dict]:

        phi_embed = state.phi.apply_fn(
            {"params": phi_params},
            batch.observation,
            train=train,
            rngs={"dropout": key},
        )
        if static_state.include_action:  # Optionally include action
            action_embed = state.action_encoder.apply_fn(
                {"params": action_encoder_params},
                batch.action,
                train=train,
                rngs={"dropout": key},
            )
            joint_embed = jnp.concatenate([phi_embed, action_embed], axis=-1)
        else:
            joint_embed = phi_embed
        phi_projections = state.phi_projector.apply_fn(
            {"params": phi_projector_params},
            joint_embed,
            train=train,
            rngs={"dropout": key},
        )

        if static_state.share_encoder:  # Optionally re-use phi encoder
            psi_embed = state.phi.apply_fn(
                {"params": phi_params},
                batch.next_observation,
                train=train,
                rngs={"dropout": key},
            )
        else:  # Otherwise use separate psi encoder
            psi_embed = state.psi.apply_fn(
                {"params": psi_params},
                batch.next_observation,
                train=train,
                rngs={"dropout": key},
            )
        # Always use distinct psi projector
        psi_projections = state.psi_projector.apply_fn(
            {"params": psi_projector_params},
            psi_embed,
            train=train,
            rngs={"dropout": key},
        )

        if static_state.label_smoothing > 0.0:  # Optionally regularize the loss
            num_permute = int(phi_projections.shape[0] * static_state.label_smoothing)
            permuted_indices = jax.random.permutation(key, jnp.arange(num_permute))
            phi_projections = jax.lax.dynamic_update_slice(
                phi_projections,
                phi_projections[permuted_indices, :],
                (0, 0),
            )

        # Compute inner product energy.
        product = (
            phi_projections[:, jnp.newaxis, :] * psi_projections[jnp.newaxis, :, :]
        )
        log_energies = jnp.sum(product, axis=-1) / static_state.temperature
        # Debugging quantities:
        accuracy = jnp.mean(jnp.diag(log_energies) >= jnp.max(log_energies, axis=-1))

        # Compute loss
        num_loss = -1.0 * jnp.diag(log_energies)
        den_loss = jax.nn.logsumexp(log_energies, axis=-1)
        rep_loss = jnp.mean(num_loss + den_loss)

        # Debugging info. No gradients passed to the encoder.
        detached_embedding = {"state": jax.lax.stop_gradient(phi_embed)}
        action_preds = state.policy.apply_fn(
            {"params": policy_params},
            detached_embedding,
            train=train,
            rngs={"dropout": key},
        )
        if len(batch.action.shape) == 2:
            policy_loss = jnp.mean((action_preds - batch.action) ** 2)
        elif len(batch.action.shape) == 3:  # Only predict first of stacked actions
            policy_loss = jnp.mean((action_preds - batch.action[:, :, 0]) ** 2)
        else:
            raise ValueError("Invalid action shape.")
        target_pred = state.target_predictor.apply_fn(
            {"params": target_predictor_params},
            jax.lax.stop_gradient(phi_embed),
            train=train,
            rngs={"dropout": key},
        )
        target_loss = jnp.mean((target_pred - batch.observation["target"]) ** 2)

        loss = rep_loss + policy_loss + target_loss
        return loss, {
            "loss": loss,
            "rep_loss": rep_loss,
            "policy_loss": policy_loss,
            "target_loss": target_loss,
            "num_loss": jnp.mean(num_loss),
            "den_loss": jnp.mean(den_loss),
            "accuracy": accuracy,
            "embedding_norm": jnp.linalg.norm(phi_projections[0]),
        }

    argnums = (0, 1, 2, 3, 4, 5, 6)
    if static_state.include_action:
        action_encoder_params = state.action_encoder.params
    else:
        argnums = argnums[:-1]
        action_encoder_params = None
    grad_fn = jax.grad(loss_fn, argnums=argnums, has_aux=True)
    grads, info = grad_fn(
        state.phi.params,
        state.phi_projector.params,
        state.psi.params,
        state.psi_projector.params,
        state.policy.params,
        state.target_predictor.params,
        action_encoder_params,
    )
    info = dict(info, grad_norm=grad_norm(grads))
    return (grads, info, rng)


def update(
    state: JointTrainState, batch: Batch, rng: PRNGKey, static_state: StaticState
) -> Tuple[TrainState, Dict, PRNGKey]:
    """Update actor parameters using MSE loss."""
    grads, info, rng = evaluate_loss(state, batch, rng, static_state, train=True)
    new_phi = state.phi.apply_gradients(grads=grads[0])
    new_phi_projector = state.phi_projector.apply_gradients(grads=grads[1])
    new_psi = state.psi.apply_gradients(grads=grads[2])
    new_psi_projector = state.psi_projector.apply_gradients(grads=grads[3])
    new_policy = state.policy.apply_gradients(grads=grads[4])
    new_target_predictor = state.target_predictor.apply_gradients(grads=grads[5])
    if static_state.include_action:
        new_action_encoder = state.action_encoder.apply_gradients(grads=grads[6])
    else:
        new_action_encoder = state.action_encoder
    new_state = state.replace(
        phi=new_phi,
        phi_projector=new_phi_projector,
        psi=new_psi,
        psi_projector=new_psi_projector,
        policy=new_policy,
        target_predictor=new_target_predictor,
        action_encoder=new_action_encoder,
    )
    return (new_state, info, rng)


_update_jit = jax.jit(update, static_argnames=("static_state",))
_evaluate_loss_jit = jax.jit(evaluate_loss, static_argnames=("static_state",))


class RepLearner(BaseLearner):
    """Contrastive representation learner."""

    def __init__(self, config: Dict, batch: Batch, fixed_encoder=None):
        super().__init__(config, batch, fixed_encoder)
        obs, self.rng = self.observation_adapter(
            self.rng, batch.observation, train=False
        )

        phi, _ = self.init_model(
            config["phi_network_name"],
            obs,
            init_args={},
            # freeze_model=config.get("freeze_encoder", False),
        )
        phi_embed = phi.apply_fn({"params": phi.params}, obs)
        if config["include_action"]:
            action_encoder, _ = self.init_model(
                config["action_encoder_network_name"], batch.action, init_args={}
            )
            action_embed = action_encoder.apply_fn(
                {"params": action_encoder.params}, batch.action
            )
            joint_embed = jnp.concatenate([phi_embed, action_embed], axis=-1)
        else:
            action_encoder = None
            joint_embed = phi_embed
        phi_proj, _ = self.init_model(
            config["phi_proj_network_name"], joint_embed, init_args={}
        )

        psi, _ = self.init_model(config["psi_network_name"], obs, init_args={})
        psi_embed = psi.apply_fn({"params": psi.params}, obs)
        psi_proj, _ = self.init_model(
            config["psi_proj_network_name"], psi_embed, init_args={}
        )

        policy, _ = self.init_model(
            config["policy_network_name"],
            {"state": psi_embed},
            init_args={"action_shape": self.action_shape},
        )
        target_predictor, _ = self.init_model(
            config["target_predictor_network_name"],
            phi_embed,
            init_args={"output_dim": obs["target"].shape[-1]},
        )

        self.state = JointTrainState(
            phi=phi,
            phi_projector=phi_proj,
            psi=psi,
            psi_projector=psi_proj,
            policy=policy,
            target_predictor=target_predictor,
            action_encoder=action_encoder,
        )
        self.static_state = StaticState(
            temperature=config["temperature"],
            label_smoothing=config["label_smoothing"],
            share_encoder=config["share_encoder"],
            include_action=config["include_action"],
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
        embed, self.rng = apply_jit(self.state.phi, adapted_obs, self.rng)
        return embed
