from typing import Any
import flax
from imitation_pretraining.networks.registry import registry

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
