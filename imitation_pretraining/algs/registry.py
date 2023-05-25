"""Algorithm registry."""
from typing import Callable
from imitation_pretraining.algs.bc import bc_learner
from imitation_pretraining.algs.rep_learning import (
    contrastive_learner,
    inverse_dynamics_learner,
    reconstruction_learner,
)
from imitation_pretraining.algs.pretrained_models import r3m, timm
from imitation_pretraining.algs.base import BaseLearner


class Registry(object):
    """A registry for agents."""

    def __init__(self):
        self._algs = {}

    def register(self, name: str, alg_constructor_fn: Callable):
        """Register an agent."""
        self._algs[name] = alg_constructor_fn
        return

    def make(self, name: str, **kwargs) -> BaseLearner:
        """Build an agent."""
        return self._algs[name](**kwargs)


registry = Registry()


registry.register("bc", bc_learner.BCLearner)
registry.register("contrastive", contrastive_learner.RepLearner)
registry.register("inverse_dynamics", inverse_dynamics_learner.RepLearner)
registry.register("reconstruction", reconstruction_learner.RepLearner)
registry.register("r3m", r3m.R3MLearner)
registry.register("timm", timm.TIMMLearner)
