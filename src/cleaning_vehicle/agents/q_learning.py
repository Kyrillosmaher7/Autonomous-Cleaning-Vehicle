from __future__ import annotations

import random
from collections import defaultdict

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.state import State

QValueTable = dict[tuple[State, Action], float]


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Uses epsilon-greedy action selection.

    Parameters
    ----------
    learning_rate:
        Alpha (α), controls how strongly new information
        changes the existing Q-value.

    discount_factor:
        Gamma (γ), controls the importance of future rewards.

    epsilon:
        Probability of choosing a random action.

    epsilon_decay:
        Multiplicative decay applied after each training episode.

    epsilon_min:
        Lower bound for epsilon.
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
            raise ValueError(
                "learning_rate must be in (0, 1]."
            )

        if not 0 <= discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in [0, 1]."
            )

        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must be in [0, 1]."
            )

        if not 0 < epsilon_decay <= 1:
            raise ValueError(
                "epsilon_decay must be in (0, 1]."
            )

        if not 0 <= epsilon_min <= 1:
            raise ValueError(
                "epsilon_min must be in [0, 1]."
            )

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._random = random.Random(seed)

        self.q_table: defaultdict[
            tuple[State, Action],
            float,
        ] = defaultdict(float)

    # ==========================================================
    # Q-value access
    # ==========================================================

    def get_q_value(
        self,
        state: State,
        action: Action,
    ) -> float:
        """Return Q(state, action)."""

        return self.q_table[(state, action)]

    # ==========================================================
    # Best action
    # ==========================================================

    def best_action(self, state: State) -> Action:
        """
        Return the action with the highest Q-value.

        Ties are resolved deterministically using Action order.
        """

        actions = list(Action)

        return max(
            actions,
            key=lambda action: self.get_q_value(
                state,
                action,
            ),
        )

    # ==========================================================
    # Epsilon-greedy policy
    # ==========================================================

    def select_action(self, state: State) -> Action:
        """
        Select an action using epsilon-greedy exploration.

        With probability epsilon:
            random action

        Otherwise:
            best known action
        """

        if self._random.random() < self.epsilon:
            return self._random.choice(list(Action))

        return self.best_action(state)

    # ==========================================================
    # Q-Learning update
    # ==========================================================

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """
        Update Q(state, action).

        Q-learning equation:

        Q(s,a) ← Q(s,a) +
            α [r + γ max Q(s',a') - Q(s,a)]

        For terminal states:

            target = reward
        """

        current_q = self.get_q_value(
            state,
            action,
        )

        if done:
            target = reward

        else:
            max_next_q = max(
                self.get_q_value(
                    next_state,
                    next_action,
                )
                for next_action in Action
            )

            target = (
                reward
                + self.discount_factor * max_next_q
            )

        new_q = current_q + (
            self.learning_rate
            * (target - current_q)
        )

        self.q_table[(state, action)] = new_q

    # ==========================================================
    # Exploration decay
    # ==========================================================

    def decay_epsilon(self) -> None:
        """Decay epsilon while respecting epsilon_min."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # ==========================================================
    # Q-table information
    # ==========================================================

    @property
    def table_size(self) -> int:
        """Return the number of explicitly stored Q-values."""

        return len(self.q_table)