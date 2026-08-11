from __future__ import annotations

from abc import ABC, abstractmethod

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.state import State


class BaseAgent(ABC):
    """
    Abstract base class for all cleaning vehicle agents.

    Every agent must implement action selection:

        State -> Action

    Future agents such as RandomAgent, GreedyAgent,
    QLearningAgent, and SARSAAgent will use this interface.
    """

    @abstractmethod
    def select_action(self, state: State) -> Action:
        """
        Select an action based on the current state.
        """
        raise NotImplementedError