from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Metrics collected during Q-Learning training."""

    episode_rewards: tuple[float, ...]
    episode_steps: tuple[int, ...]
    episode_cleaned_cells: tuple[int, ...]
    episode_completions: tuple[bool, ...]


def _train_step(
    environment: CleaningEnvironment,
    agent: QLearningAgent,
    state: State,
    use_observations: bool,
) -> tuple[State, float, bool]:
    """Run one learning step and return (next_state, reward, done)."""

    if use_observations:
        observation = environment.get_observation(state)
        action = agent.select_observation(observation)

        next_state, reward, done = environment.step(state, action)

        agent.update_observation(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=environment.get_observation(next_state),
            done=done,
        )
    else:
        action = agent.select_action(state)

        next_state, reward, done = environment.step(state, action)

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    return next_state, reward, done


def train_q_learning(
    environment: CleaningEnvironment,
    agent: QLearningAgent,
    initial_state_factory: Callable[[], State],
    episodes: int = 1000,
    max_steps: int = 100,
    use_observations: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """
    Train a Q-Learning agent.

    Parameters
    ----------
    environment:
        Environment to train in.
    agent:
        QLearningAgent being trained.
    initial_state_factory:
        Returns the initial State for each episode.
    episodes:
        Number of training episodes.
    max_steps:
        Maximum number of steps per episode.
    use_observations:
        False (default): learn from the full State (V1).
        True: learn from partial Observations (V2).
    progress_callback:
        Optional ``callback(completed_episodes, total_episodes)``
        called after every episode.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    rewards: list[float] = []
    steps_per_episode: list[int] = []
    cleaned: list[int] = []
    completions: list[bool] = []

    for episode in range(1, episodes + 1):

        state = initial_state_factory()
        initial_dirty_count = len(state.dirty_cells)

        total_reward = 0.0
        steps = 0

        while steps < max_steps:

            state, reward, done = _train_step(
                environment, agent, state, use_observations
            )

            total_reward += reward
            steps += 1

            if done:
                break

        rewards.append(total_reward)
        steps_per_episode.append(steps)
        cleaned.append(initial_dirty_count - len(state.dirty_cells))
        completions.append(len(state.dirty_cells) == 0)

        agent.decay_epsilon()

        if progress_callback is not None:
            progress_callback(episode, episodes)

    return TrainingResult(
        episode_rewards=tuple(rewards),
        episode_steps=tuple(steps_per_episode),
        episode_cleaned_cells=tuple(cleaned),
        episode_completions=tuple(completions),
    )
