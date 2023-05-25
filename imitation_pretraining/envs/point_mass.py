# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np


DEFAULT_TIME_LIMIT = 5
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    cwd = os.path.dirname(__file__)
    return resources.GetResource(os.path.join(cwd, "my_point_mass.xml")), common.ASSETS


@SUITE.add()
def goal_based(
    time_limit=DEFAULT_TIME_LIMIT,
    random=None,
    task_id=None,
    vis_goal=False,
    noise_scale=0.0,
    environment_kwargs=None,
):
    """Returns the goal_based point_mass task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = GoalBasedPointMass(
        randomize_gains=False,
        task_id=task_id,
        random=random,
        vis_goal=vis_goal,
        noise_scale=noise_scale,
    )
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""

    def mass_to_target(self):
        """Returns the vector from mass to target in global coordinate."""
        return (
            self.named.data.geom_xpos["target"] - self.named.data.geom_xpos["pointmass"]
        )

    def mass_to_target_dist(self):
        """Returns the distance from mass to the target."""
        return np.linalg.norm(self.mass_to_target())

    def mass_to_goal(self, goal):
        """Returns the vector from mass to goal in global coordinate."""
        return goal - self.named.data.geom_xpos["pointmass"]

    def mass_to_goal_dist(self, goal):
        """Returns the distance from mass to the target."""
        return np.linalg.norm(self.mass_to_goal(goal))


# pylint: disable=abstract-method
class PointMass(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, randomize_gains, random=None):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        super().__init__(random=random)
        self._success = 0.0

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

           If _randomize_gains is True, the relationship between the controls and
           the joints is randomized, so that each control actuates a random linear
           combination of joints.

        Args:
          physics: An instance of `mujoco.Physics`.
        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        if self._randomize_gains:
            dir1 = self.random.randn(2)
            dir1 /= np.linalg.norm(dir1)
            # Find another actuation direction that is not 'too parallel' to dir1.
            parallel = True
            while parallel:
                dir2 = self.random.randn(2)
                dir2 /= np.linalg.norm(dir2)
                parallel = abs(np.dot(dir1, dir2)) > 0.9
            physics.model.wrap_prm[[0, 1]] = dir1
            physics.model.wrap_prm[[2, 3]] = dir2
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["velocity"] = physics.velocity()
        obs["success"] = np.array([self._success], dtype=np.float32)
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        target_size = physics.named.model.geom_size["target", 0]
        near_target = rewards.tolerance(
            physics.mass_to_target_dist(), bounds=(0, target_size), margin=target_size
        )
        control_reward = rewards.tolerance(
            physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        small_control = (control_reward + 4) / 5
        reward = near_target * small_control
        self._success = 1.0 if near_target > 0.99 else 0.0
        return reward


class GoalBasedPointMass(PointMass):
    """A goal-based variant of the PointMass task."""

    def __init__(
        self,
        randomize_gains,
        task_id=None,
        vis_goal=False,
        random=None,
        noise_scale=0.0,
    ):
        """Initialize an instance of `GoalBasedPointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(randomize_gains=randomize_gains, random=random)

        self.task_id = task_id
        if task_id is not None:
            task_rng = np.random.default_rng(task_id)
            self.goal = self._generate_goal(task_rng)
        else:
            self.goal = self._generate_goal(self._random)
        self.vis_goal = vis_goal
        self.noise_scale = noise_scale

    def _generate_goal(self, rng):
        """Generates a new goal."""
        return np.concatenate([rng.uniform(-0.29, 0.29, size=2), np.array([0.01])])

    def initialize_episode(self, physics):
        if self.task_id is None:
            self.goal = self._generate_goal(self._random)
        # Set goal in physics
        physics.named.model.geom_pos["target"] = self.goal
        # Hide goal if not visualizing
        if not self.vis_goal:
            physics.named.model.geom_rgba["target", -1] = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs["goal"] = self.goal[:2]
        return obs

    def before_step(self, action, physics):
        """Applies noise to qpos before stepping the simulator."""
        super().before_step(action, physics)
        physics.data.qpos += self.random.normal(
            scale=self.noise_scale, size=physics.data.qpos.shape
        )
