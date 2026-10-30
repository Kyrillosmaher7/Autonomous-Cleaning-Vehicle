from __future__ import annotations

from dataclasses import dataclass

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """
    Metrics collected from one episode.
    """

    total_reward: float
    steps: int
    cleaned_cells: int
    completed: bool
    invalid_actions: int


def run_episode(
    environment: CleaningEnvironment,
    agent: BaseAgent,
    initial_state: State,
    max_steps: int,
) -> EpisodeResult:
    """
    Run one episode using an agent.

    The agent selects actions.
    The environment executes actions.

    Transition:

        State
          ↓
        Agent
          ↓
        Action
          ↓
        Environment
          ↓
        Next State + Reward + Done

    Note:
        `done` means the environment episode has terminated.

        `completed` specifically means that all dirty cells
        were successfully cleaned.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    state = initial_state

    initial_dirty_count = len(initial_state.dirty_cells)

    total_reward = 0.0
    steps = 0
    invalid_actions = 0

    while (
        len(state.dirty_cells) > 0
        and steps < max_steps
    ):
        action = agent.select_action(state)

        if not isinstance(action, Action):
            raise TypeError(
                "agent.select_action() must return an Action"
            )

        next_state, reward, done = environment.step(
            state,
            action,
        )

        if next_state.position == state.position:
            invalid_actions += 1

        total_reward += reward
        steps += 1

        state = next_state

        # The environment can terminate because of:
        #   - successful completion
        #   - battery depletion
        #
        # We must stop the episode in either case.
        if done:
            break

    cleaned_cells = (
        initial_dirty_count
        - len(state.dirty_cells)
    )

    # IMPORTANT:
    # Completion means ALL dirt was cleaned.
    # Battery depletion is not successful completion.
    completed = len(state.dirty_cells) == 0

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        cleaned_cells=cleaned_cells,
        completed=completed,
        invalid_actions=invalid_actions,
    )