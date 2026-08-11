from __future__ import annotations

import random

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.state import State

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.

    This agent does not learn.

    It exists primarily as a baseline against which
    future agents can be compared.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def select_action(self, state: State) -> Action:
        """
        Select one of the available actions uniformly.
        """
        del state

        return self._random.choice(tuple(Action))