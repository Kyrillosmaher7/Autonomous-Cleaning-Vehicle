from __future__ import annotations

from dataclasses import dataclass

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Metrics collected during Q-Learning training."""

    episode_rewards: tuple[float, ...]
    episode_steps: tuple[int, ...]
    episode_cleaned_cells: tuple[int, ...]
    episode_completions: tuple[bool, ...]


def train_q_learning(
    environment: CleaningEnvironment,
    agent: QLearningAgent,
    initial_state_factory,
    episodes: int = 1000,
    max_steps: int = 100,
) -> TrainingResult:
    """
    Train a Q-Learning agent.

    Each episode:

        initial state
            ↓
        select action
            ↓
        environment.step()
            ↓
        Q-value update
            ↓
        repeat until done
            ↓
        decay epsilon
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_cleaned_cells: list[int] = []
    episode_completions: list[bool] = []

    for _ in range(episodes):
        state: State = initial_state_factory()

        initial_dirty_count = len(state.dirty_cells)

        total_reward = 0.0
        steps = 0

        while steps < max_steps:
            action = agent.select_action(state)

            next_state, reward, done = environment.step(
                state,
                action,
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            total_reward += reward
            steps += 1

            state = next_state

            if done:
                break

        cleaned_cells = (
            initial_dirty_count
            - len(state.dirty_cells)
        )

        completed = len(state.dirty_cells) == 0

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_cleaned_cells.append(cleaned_cells)
        episode_completions.append(completed)

        agent.decay_epsilon()

    return TrainingResult(
        episode_rewards=tuple(episode_rewards),
        episode_steps=tuple(episode_steps),
        episode_cleaned_cells=tuple(episode_cleaned_cells),
        episode_completions=tuple(episode_completions),
    )