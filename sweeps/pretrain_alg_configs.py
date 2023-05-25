"""Configs for pretrain algs."""
from sweep_utils import grid_to_list


def alg_configs(identity_encoder=False):
    """Configs for pretrain algs.
    5 algs."""
    config_list = []

    if identity_encoder:
        enc_name = "identity"
    else:
        enc_name = "conv"

    # BC
    alg_grid = dict(
        agent_name=["bc"],
        predict_target=[True],
        policy_network_name=[f"policy-{enc_name}-softmax-64-norm-0.1"],
    )
    config_list.extend(grid_to_list(alg_grid))

    # Inverse Dynamics
    alg_grid = dict(
        agent_name=["inverse_dynamics"],
        phi_network_name=[f"encoder-{enc_name}-softmax-64-norm"],
        policy_network_name=["policy-mlp-256-0.1"],
        predict_all_actions=[False],
        nstep=[1],
    )
    config_list.extend(grid_to_list(alg_grid))

    # Contrastive
    cont_grid = dict(
        agent_name=["contrastive"],
        phi_network_name=[f"encoder-{enc_name}-softmax-64-norm"],
        phi_projector_network_name=["encoder-projector-64-0.0"],
        psi_network_name=[f"encoder-{enc_name}-softmax-64-norm"],
        psi_projector_network_name=["encoder-projector-64-0.0"],
        action_encoder_network_name=["encoder-action-64-0.0"],
        policy_network_name=["policy-mlp-256-0.1"],
        temperature=[1.0],
    )

    # SimCLR
    alg_grid = dict(
        cont_grid,
        include_action=[False],
        share_encoder=[True],
        nstep=[0],
    )
    config_list.extend(grid_to_list(alg_grid))

    # Contrastive dynamics
    alg_grid = dict(
        cont_grid,
        include_action=[True],
        share_encoder=[True],
        nstep=[1],
    )
    config_list.extend(grid_to_list(alg_grid))

    # Reconstruction dynamics
    alg_grid = dict(
        agent_name=["reconstruction"],
        encoder_network_name=[f"encoder-{enc_name}-softmax-64-norm"],
        decoder_network_name=[f"decoder-conv-120"],  # Default to larger decoder
        action_encoder_network_name=["encoder-action-64-0.0"],
        policy_network_name=["policy-mlp-256-0.1"],
        include_action=[True],
        nstep=[1],
    )
    config_list.extend(grid_to_list(alg_grid))

    return config_list


def sweep_nstep_alg_configs():
    """Configs for pretrain algs.
    6 algs."""
    config_list = []

    # Inverse Dynamics
    alg_grid = dict(
        agent_name=["inverse_dynamics"],
        phi_network_name=["encoder-conv-softmax-64-norm"],
        policy_network_name=["policy-mlp-256-0.1"],
        predict_all_actions=[False],
        nstep=[2, 5, 10],
    )
    config_list.extend(grid_to_list(alg_grid))

    # Contrastive
    alg_grid = dict(
        agent_name=["contrastive"],
        phi_network_name=["encoder-conv-softmax-64-norm"],
        phi_projector_network_name=["encoder-projector-64-0.0"],
        psi_network_name=["encoder-conv-softmax-64-norm"],
        psi_projector_network_name=["encoder-projector-64-0.0"],
        action_encoder_network_name=["encoder-action-64-0.0"],
        policy_network_name=["policy-mlp-256-0.1"],
        temperature=[1.0],
        include_action=[True],
        share_encoder=[True],
        nstep=[2, 5, 10],
    )
    config_list.extend(grid_to_list(alg_grid))

    return config_list
