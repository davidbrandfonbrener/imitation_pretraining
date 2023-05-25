import bc, contrastive, inverse_dynamics, reconstruction


def get_config(agent_name, local=False):
    """Returns the default config for CD pretraining."""
    if agent_name == "bc":
        config = bc.get_config(local)
    elif agent_name == "contrastive":
        config = contrastive.get_config(local)
    elif agent_name == "inverse_dynamics":
        config = inverse_dynamics.get_config(local)
    elif agent_name == "reconstruction":
        config = reconstruction.get_config(local)
    else:
        raise ValueError(f"Agent {agent_name} not supported.")
    return config
