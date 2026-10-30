"""
Headless simulation used by the dashboard.

All stepping logic lives here so the GUI only draws. One class serves
every environment version: partial-observation versions feed the agent
an Observation, full-state versions feed it the State.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.observation import Observation
from cleaning_vehicle.environment.state import State

END_COMPLETED = "completed"
END_BATTERY = "battery depleted"
END_TIME_LIMIT = "time limit reached"


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """What happened in one simulation step."""

    action: Action
    reward: float
    cleaned: bool
    done: bool


class Simulation:
    """One agent acting in one environment, one step at a time."""

    def __init__(
        self,
        environment_factory: Callable[[], CleaningEnvironment],
        state_factory: Callable[[], State],
        agent: BaseAgent,
        partial_observation: bool,
        max_steps: int,
    ) -> None:

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self._environment_factory = environment_factory
        self._state_factory = state_factory
        self.agent = agent
        self.partial_observation = partial_observation
        self.max_steps = max_steps

        self.reset()

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def reset(self) -> None:
        """
        Restart the episode.

        The environment is rebuilt as well, because dynamic obstacles
        and the random generator are part of its mutable state.
        """

        self.environment = self._environment_factory()
        self.state = self._state_factory()

        self.step_count = 0
        self.total_reward = 0.0
        self.cleaned_count = 0
        self.last_outcome: StepOutcome | None = None
        self.end_reason: str | None = None

        self._check_finished(done=False)

    @property
    def finished(self) -> bool:
        return self.end_reason is not None

    @property
    def observation(self) -> Observation | None:
        """Current partial observation, or None in full-state mode."""

        if not self.partial_observation:
            return None

        return self.environment.get_observation(self.state)

    # ----------------------------------------------------------
    # Stepping
    # ----------------------------------------------------------

    def _choose_action(self) -> Action:
        if self.partial_observation:
            return self.agent.select_observation(self.observation)

        return self.agent.select_action(self.state)

    def _check_finished(self, done: bool) -> None:
        if done:
            self.end_reason = (
                END_BATTERY if self.state.battery == 0 else END_COMPLETED
            )
        elif self.step_count >= self.max_steps:
            self.end_reason = END_TIME_LIMIT

    def step(self) -> StepOutcome | None:
        """Advance one step. Returns None once the episode is over."""

        if self.finished:
            return None

        action = self._choose_action()

        next_state, reward, done = self.environment.step(self.state, action)

        # Dirt can spawn in the same step it is cleaned, so compare
        # the destination cell instead of the dirty-set sizes.
        cleaned = next_state.position in self.state.dirty_cells

        self.state = next_state
        self.step_count += 1
        self.total_reward += reward
        self.cleaned_count += int(cleaned)

        self.last_outcome = StepOutcome(action, reward, cleaned, done)
        self._check_finished(done)

        return self.last_outcome

    # ----------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------

    def summary(self) -> str:
        reason = self.end_reason or "running"

        return (
            f"{reason} | steps={self.step_count} "
            f"reward={self.total_reward:.1f} "
            f"cleaned={self.cleaned_count} "
            f"battery={self.state.battery} "
            f"dirty_left={len(self.state.dirty_cells)}"
        )
