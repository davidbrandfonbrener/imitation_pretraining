"""Contrastive representation learning using dynamics."""
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
    """Joint training state for encoder-decoder model."""

    encoder: TrainState
    action_encoder: TrainState
    decoder: TrainState
    policy: TrainState  # for debugging purposes
    target_predictor: TrainState  # for debugging purposes


class StaticState(struct.PyTreeNode):
    """Static state for encoder-decoder model."""

    predict_pixels: bool = True


def evaluate_loss(
    state: JointTrainState,
    batch: Batch,
    rng: PRNGKey,
    static_state: StaticState,
    train=False,
) -> Tuple[Tuple[Params], Dict, PRNGKey]:
    """Evaluate reconstruction loss."""
    rng, key = jax.random.split(rng)

    def loss_fn(
        encoder_params: Params,
        action_encoder_params: Params,
        decoder_params: Params,
        policy_params: Params,
        target_predictor_params: Params,
    ) -> Tuple[jnp.ndarray, Dict]:

        embed = state.encoder.apply_fn(
            {"params": encoder_params},
            batch.observation,
            train=train,
            rngs={"dropout": key},
        )
        action_embed = state.action_encoder.apply_fn(
            {"params": action_encoder_params},
            batch.action,
            train=train,
            rngs={"dropout": key},
        )
        joint_embed = jnp.concatenate([embed, action_embed], axis=-1)

        if static_state.predict_pixels:
            obs_key = "pixels"
        else:
            obs_key = "state"
        target = batch.next_observation[obs_key] - 0.5
        reconstructed_obs = state.decoder.apply_fn(
            {"params": decoder_params},
            joint_embed,
            train=train,
            rngs={"dropout": key},
        )
        rec_loss = jnp.mean((reconstructed_obs - target) ** 2)

        # Policy loss on stop_grad embeddings for debugging purposes
        action_preds = state.policy.apply_fn(
            {"params": policy_params},
            {"state": jax.lax.stop_gradient(embed)},
            train=train,
            rngs={"dropout": key},
        )
        if len(batch.action.shape) == 2:
            policy_loss = jnp.mean((action_preds - batch.action) ** 2)
        elif len(batch.action.shape) == 3:  # Only predict first of stacked actions
            policy_loss = jnp.mean((action_preds - batch.action[:, :, 0]) ** 2)
        else:
            raise ValueError("Invalid action shape.")
        # Predict target for debugging purposes
        target_pred = state.target_predictor.apply_fn(
            {"params": target_predictor_params},
            jax.lax.stop_gradient(embed),
            train=train,
            rngs={"dropout": key},
        )
        target_loss = jnp.mean((target_pred - batch.observation["target"]) ** 2)

        loss = rec_loss + policy_loss + target_loss
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "target_loss": target_loss,
            "rec_loss": rec_loss,
        }

    argnums = (0, 1, 2, 3, 4)
    grad_fn = jax.grad(loss_fn, argnums=argnums, has_aux=True)
    grads, info = grad_fn(
        state.encoder.params,
        state.action_encoder.params,
        state.decoder.params,
        state.policy.params,
        state.target_predictor.params,
    )
    info = dict(info, grad_norm=grad_norm(grads))
    return (grads, info, rng)


def update(
    state: JointTrainState, batch: Batch, rng: PRNGKey, static_state: StaticState
) -> Tuple[JointTrainState, Dict, PRNGKey]:
    """Update state."""
    grads, info, rng = evaluate_loss(state, batch, rng, static_state, train=True)
    new_encoder = state.encoder.apply_gradients(grads=grads[0])
    new_action_encoder = state.action_encoder.apply_gradients(grads=grads[1])
    new_decoder = state.decoder.apply_gradients(grads=grads[2])
    new_policy = state.policy.apply_gradients(grads=grads[3])
    new_target_predictor = state.target_predictor.apply_gradients(grads=grads[4])
    new_state = state.replace(
        encoder=new_encoder,
        action_encoder=new_action_encoder,
        decoder=new_decoder,
        policy=new_policy,
        target_predictor=new_target_predictor,
    )
    return (new_state, info, rng)


_evaluate_loss_jit = jax.jit(evaluate_loss, static_argnames=("static_state",))
_update_jit = jax.jit(update, static_argnames=("static_state",))


class RepLearner(BaseLearner):
    """Reconstruction representation learner."""

    def __init__(self, config: Dict, batch: Batch, fixed_encoder=None) -> None:
        super().__init__(config, batch, fixed_encoder)
        obs, self.rng = self.observation_adapter(
            self.rng, batch.observation, train=False
        )

        encoder, _ = self.init_model(
            config["encoder_network_name"],
            obs,
            init_args={},
            # freeze_model=config.get("freeze_encoder", False),
        )
        embed = encoder.apply_fn({"params": encoder.params}, obs)
        action_encoder, _ = self.init_model(
            config["action_encoder_network_name"], batch.action, init_args={}
        )
        action_embed = action_encoder.apply_fn(
            {"params": action_encoder.params}, batch.action
        )
        joint_embed = jnp.concatenate([embed, action_embed], axis=-1)
        decoder, _ = self.init_model(config["decoder_network_name"], joint_embed, {})
        policy, _ = self.init_model(
            config["policy_network_name"],
            {"state": embed},
            {"action_shape": self.action_shape},
        )
        target_predictor, _ = self.init_model(
            config["target_predictor_network_name"],
            embed,
            init_args={"output_dim": obs["target"].shape[-1]},
        )
        self.state = JointTrainState(
            encoder=encoder,
            action_encoder=action_encoder,
            decoder=decoder,
            policy=policy,
            target_predictor=target_predictor,
        )
        self.static_state = StaticState(predict_pixels=config["predict_pixels"])

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
        return self.state.encoder.params

    def encode(self, observation: Dict) -> jnp.ndarray:
        adapted_obs, self.rng = self.observation_adapter(
            self.rng, observation, train=False
        )
        output, self.rng = apply_jit(self.state.encoder, adapted_obs, self.rng)
        return output

    def autoencode(self, observation: Dict, action: jnp.ndarray = None) -> jnp.ndarray:
        """Autoencode observation to pixels for debugging."""
        adapted_obs, self.rng = self.observation_adapter(
            self.rng, observation, train=False
        )
        embed, self.rng = apply_jit(self.state.encoder, adapted_obs, self.rng)
        if self.config["include_action"]:
            assert action is not None, "Action must be provided"
            action_embed, self.rng = apply_jit(
                self.state.action_encoder, action, self.rng
            )
            embed = jnp.concatenate([embed, action_embed], axis=-1)
        # (mu, logvar), self.rng = apply_jit(
        #     self.state.encoder_projector, embed, self.rng
        # )
        # if sample:
        #     latent = mu + jnp.exp(0.5 * logvar) * jax.random.normal(
        #         self.rng, logvar.shape
        #     )
        # else:
        #     latent = mu
        output, self.rng = apply_jit(self.state.decoder, embed, self.rng)
        return output
