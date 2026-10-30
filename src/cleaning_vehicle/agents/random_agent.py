from __future__ import annotations

import random

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.observation import Observation
from cleaning_vehicle.environment.state import State

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.

    Supports both V1 full-state input and V2 partial-observation
    input.

    The agent does not use either input when selecting an action.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def select_action(self, state: State) -> Action:
        """
        V1 interface.

        Select one of the available actions uniformly.
        """

        del state

        return self._random.choice(tuple(Action))

    def select_observation(
        self,
        observation: Observation,
    ) -> Action:
        """
        V2 interface.

        Select one of the available actions uniformly from
        a partial observation.

        The observation is intentionally ignored because this
        is a random baseline agent.
        """

        del observation

        return self._random.choice(tuple(Action))