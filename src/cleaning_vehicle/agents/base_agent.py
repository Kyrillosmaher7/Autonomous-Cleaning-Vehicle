from __future__ import annotations

from abc import ABC, abstractmethod

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.observation import Observation
from cleaning_vehicle.environment.state import State


class BaseAgent(ABC):
    """
    Abstract base class for cleaning vehicle agents.

    Two input styles exist, matching the two environment modes:

        full state (V1):           select_action(state)
        partial observation (V2):  select_observation(observation)

    Every agent must support the full-state interface. Agents that can
    work from a partial observation override ``select_observation``.
    """

    @abstractmethod
    def select_action(self, state: State) -> Action:
        """Select an action from the complete environment state."""

    def select_observation(self, observation: Observation) -> Action:
        """Select an action from a partial observation."""

        del observation

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support "
            "partial observations."
        )
