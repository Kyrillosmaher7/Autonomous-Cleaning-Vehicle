from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.observation import Observation
from cleaning_vehicle.environment.state import State

from .base_agent import BaseAgent

ACTIONS: tuple[Action, ...] = tuple(Action)


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.

    Two independent Q-tables are kept, one per environment mode:

        V1: ``q_table``             keyed by (State, Action)
        V2: ``observation_q_table`` keyed by (Observation, Action)

    Both use the same update rule:

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))

    and the same epsilon-greedy policy. Hyperparameters normally come
    from ``configs/agents/q_learning.yaml``.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ) -> None:

        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")

        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1].")

        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")

        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1].")

        if not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon_min must be in [0, 1].")

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Instance-local RNG keeps seeded runs reproducible.
        self._random = random.Random(seed)

        self.q_table: defaultdict[tuple[State, Action], float] = (
            defaultdict(float)
        )
        self.observation_q_table: defaultdict[
            tuple[Observation, Action], float
        ] = defaultdict(float)

    # ==========================================================
    # Shared helpers
    # ==========================================================

    def _best_actions(self, q_values: dict[Action, float]) -> list[Action]:
        """All actions tied for the highest Q-value."""
        best = max(q_values.values())
        return [a for a in ACTIONS if q_values[a] == best]

    def _td_update(
        self,
        table: defaultdict,
        key: tuple[Hashable, Action],
        reward: float,
        next_max_q: float,
        done: bool,
    ) -> None:
        """Apply one Q-learning update to ``table[key]``."""

        current_q = table[key]
        target = reward if done else (
            reward + self.discount_factor * next_max_q
        )

        table[key] = current_q + self.learning_rate * (target - current_q)

    def _explore(self) -> bool:
        return self._random.random() < self.epsilon

    # ==========================================================
    # V1 - State based
    # ==========================================================

    def get_q_value(self, state: State, action: Action) -> float:
        """Return Q(state, action)."""
        return self.q_table[(state, action)]

    def best_action(self, state: State) -> Action:
        """
        Action with the highest Q-value.

        Ties are resolved deterministically by Action order.
        """
        return max(ACTIONS, key=lambda a: self.get_q_value(state, a))

    def select_action(self, state: State) -> Action:
        """Epsilon-greedy action from the complete State."""

        if self._explore():
            return self._random.choice(ACTIONS)

        return self.best_action(state)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Q-learning update for a State transition."""

        next_max_q = 0.0 if done else max(
            self.get_q_value(next_state, a) for a in ACTIONS
        )

        self._td_update(
            self.q_table, (state, action), reward, next_max_q, done
        )

    # ==========================================================
    # V2 - Observation based
    # ==========================================================

    def get_observation_q_value(
        self,
        observation: Observation,
        action: Action,
    ) -> float:
        """
        Return Q(observation, action).

        Entries inserted manually into ``q_table`` are honoured so
        older tests that seed that table keep working.
        """

        key = (observation, action)

        if key in self.q_table:
            return self.q_table[key]

        return self.observation_q_table[key]

    def best_observation_action(self, observation: Observation) -> Action:
        """
        Action with the highest Q-value for an observation.

        Ties are broken randomly so an untrained table does not bias
        the agent toward the first Action.
        """

        q_values = {
            a: self.get_observation_q_value(observation, a)
            for a in ACTIONS
        }

        return self._random.choice(self._best_actions(q_values))

    def select_observation(self, observation: Observation) -> Action:
        """Epsilon-greedy action from a partial Observation."""

        if self._explore():
            return self._random.choice(ACTIONS)

        return self.best_observation_action(observation)

    def update_observation(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        """Q-learning update for an Observation transition."""

        next_max_q = 0.0 if done else max(
            self.get_observation_q_value(next_observation, a)
            for a in ACTIONS
        )

        self._td_update(
            self.observation_q_table,
            (observation, action),
            reward,
            next_max_q,
            done,
        )

    # ==========================================================
    # Exploration and table info
    # ==========================================================

    def decay_epsilon(self) -> None:
        """Decay epsilon, never going below epsilon_min."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    @property
    def table_size(self) -> int:
        """Number of stored V1 Q-values."""
        return len(self.q_table)

    @property
    def observation_table_size(self) -> int:
        """Number of stored V2 Q-values."""
        return len(self.observation_q_table)
