"""Registry for networks."""
import functools as ft
from typing import Dict, Tuple
from flax import linen as nn
from imitation_pretraining.networks import policy, encoder, mlp
from imitation_pretraining.networks import simple_conv


class Registry(object):
    """A registry for networks."""

    def __init__(self):
        self._networks = {}

    def register(self, name, model_init_fn):
        """Register a network."""
        self._networks[name] = model_init_fn
        return

    def make(self, name, key, *args, init_args=None) -> Tuple[nn.Module, Dict]:
        """Build and initialize a network. Return the model and params."""
        if init_args is None:
            init_args = {}
        model = self._networks[name](**init_args)
        variables = model.init(key, *args)
        print(nn.tabulate(model, key)(*args))  # Log the model structure
        return model, variables


registry = Registry()

registry.register(
    "policy-linear",
    ft.partial(
        policy.MLPPolicy,
        hidden_dims=[],
    ),
)

for width in [64, 128, 256, 512]:
    for dropout in [0.0, 0.1]:
        registry.register(
            f"policy-mlp-{width}-{dropout}",
            ft.partial(
                policy.MLPPolicy,
                hidden_dims=(width, width),
                dropout_rate=dropout,
            ),
        )

for embed in [16, 64, 128, 256]:
    for encoder_name in ["conv", "mlp", "identity"]:
        for activation in ["softmax", "embedding", "none"]:
            for normalize in [True, False]:

                if encoder_name == "conv":
                    backbone = simple_conv.SimpleConv()
                elif encoder_name == "mlp":
                    backbone = mlp.MLP(output_dim=256, hidden_dims=(256, 256))
                elif encoder_name == "identity":
                    backbone = None
                else:
                    raise ValueError("Invalid encoder name.")

                if activation == "softmax":
                    enc_args = {"spatial_softmax": 256}
                elif activation == "embedding":
                    enc_args = {"spatial_embedding": 4}
                elif activation == "none":
                    enc_args = {}
                else:
                    raise ValueError("Invalid activation name.")
                if encoder_name == "identity":
                    enc_args = {"identity": True}

                base_encoder = encoder.Encoder(
                    backbone=backbone,
                    latent_dim=embed,
                    normalize_merge=normalize,
                    **enc_args,
                )
                hidden_dims = (256, 256)
                if normalize:
                    enc_name = f"{encoder_name}-{activation}-{embed}-norm"
                else:
                    enc_name = f"{encoder_name}-{activation}-{embed}"

                for dropout in [0.0, 0.1]:
                    registry.register(
                        f"policy-{enc_name}-{dropout}",
                        ft.partial(
                            policy.EncoderPolicy,
                            encoder=base_encoder,
                            hidden_dims=hidden_dims,
                            dropout_rate=dropout,
                        ),
                    )
                    registry.register(
                        f"policy-multihead-{enc_name}-{dropout}",
                        ft.partial(
                            policy.MultiheadEncoderPolicy,
                            encoder=base_encoder,
                            hidden_dims=hidden_dims,
                            dropout_rate=dropout,
                        ),
                    )
                    registry.register(
                        f"policy-goalpixels-{enc_name}-{dropout}",
                        ft.partial(
                            policy.GoalPixelsPolicy,
                            encoder=base_encoder,
                            hidden_dims=hidden_dims,
                            dropout_rate=dropout,
                        ),
                    )

                registry.register(
                    f"encoder-{enc_name}",
                    ft.partial(
                        encoder.Encoder,
                        backbone=backbone,
                        latent_dim=embed,
                        normalize_merge=normalize,
                        **enc_args,
                    ),
                )

for embed in [16, 64, 256]:
    for dropout in [0.0, 0.1]:

        hidden_dims = (256, 256)
        # Action encoders
        registry.register(
            f"encoder-action-{embed}-{dropout}",
            ft.partial(
                encoder.ActionEncoder,
                latent_dim=embed,
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
                normalize_merge=True,
            ),
        )

        # Embedding projectors
        registry.register(
            f"encoder-projector-{embed}-{dropout}",
            ft.partial(
                encoder.Projector,
                latent_dim=embed,
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
                normalize_projection=True,
            ),
        )
        registry.register(
            f"encoder-double-projector-{embed}-{dropout}",
            ft.partial(
                encoder.DoubleProjector,
                latent_dim=embed,
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
                normalize_projection=True,
            ),
        )

        # VAE projector
        registry.register(
            f"vae-projector-{embed}-{dropout}",
            ft.partial(
                encoder.VAEProjector,
                latent_dim=embed,
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
            ),
        )

for frame_size in [84, 120]:
    registry.register(
        f"decoder-conv-{frame_size}",
        ft.partial(simple_conv.SimpleDeconv, frame_size=frame_size),
    )

    registry.register(
        f"decoder-conv-{frame_size}-large",
        ft.partial(
            simple_conv.SimpleDeconv,
            frame_size=frame_size,
            features=(256, 256, 256, 256),
        ),
    )

    registry.register(
        f"decoder-conv-{frame_size}-small",
        ft.partial(
            simple_conv.SimpleDeconv, frame_size=frame_size, features=(16, 16, 16, 16)
        ),
    )

# Note: target mlp needs to know desired output dimension
registry.register("target-mlp", ft.partial(mlp.MLP, hidden_dims=(64,)))
