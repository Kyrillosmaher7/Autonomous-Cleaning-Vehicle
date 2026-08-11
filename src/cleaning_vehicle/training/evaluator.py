from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregated metrics from evaluation episodes."""

    episodes: int
    average_reward: float
    average_steps: float
    average_cleaned_cells: float
    completion_rate: float
    average_invalid_actions: float


AgentFactory = Callable[[], BaseAgent]
StateFactory = Callable[[], State]


def evaluate_agent(
    environment: CleaningEnvironment,
    agent: BaseAgent | None = None,
    agent_factory: AgentFactory | None = None,
    initial_state_factory: StateFactory | None = None,
    episodes: int = 100,
    max_steps: int = 100,
) -> EvaluationResult:
    """
    Evaluate an agent without learning.

    Two evaluation modes are supported:

    1. Existing agent:

        evaluate_agent(
            environment=env,
            agent=trained_agent,
            ...
        )

    2. Agent factory:

        evaluate_agent(
            environment=env,
            agent_factory=RandomAgent,
            ...
        )

    When using an agent factory, a new agent is created for
    every episode.

    Q-Learning agents are evaluated greedily by temporarily
    setting epsilon to zero. The original epsilon value is
    restored after evaluation.
    """

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if agent is None and agent_factory is None:
        raise ValueError(
            "Either agent or agent_factory must be provided"
        )

    if agent is not None and agent_factory is not None:
        raise ValueError(
            "Provide either agent or agent_factory, not both"
        )

    if initial_state_factory is None:
        raise ValueError(
            "initial_state_factory must be provided"
        )

    if episodes <= 0:
        raise ValueError(
            "episodes must be positive"
        )

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be positive"
        )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    rewards: list[float] = []
    steps_list: list[int] = []
    cleaned_list: list[int] = []
    completed_list: list[bool] = []
    invalid_list: list[int] = []

    # --------------------------------------------------------------
    # Existing-agent Q-Learning evaluation
    # --------------------------------------------------------------

    original_epsilon: float | None = None

    if isinstance(agent, QLearningAgent):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

    try:

        # ==========================================================
        # Episodes
        # ==========================================================

        for _ in range(episodes):

            # ------------------------------------------------------
            # Create/select agent for this episode
            # ------------------------------------------------------

            if agent_factory is not None:
                episode_agent = agent_factory()
            else:
                # agent cannot be None here because of validation.
                episode_agent = agent

            # ------------------------------------------------------
            # Q-Learning agents created through a factory
            # also need greedy evaluation.
            # ------------------------------------------------------

            episode_original_epsilon: float | None = None

            if isinstance(episode_agent, QLearningAgent):
                episode_original_epsilon = episode_agent.epsilon
                episode_agent.epsilon = 0.0

            try:

                # --------------------------------------------------
                # Initial state
                # --------------------------------------------------

                state = initial_state_factory()

                initial_dirty = len(
                    state.dirty_cells
                )

                total_reward = 0.0
                steps = 0
                invalid_actions = 0

                done = len(
                    state.dirty_cells
                ) == 0

                # --------------------------------------------------
                # Episode loop
                # --------------------------------------------------

                while not done and steps < max_steps:

                    action = episode_agent.select_action(
                        state
                    )

                    if not isinstance(action, Action):
                        raise TypeError(
                            "agent.select_action() "
                            "must return an Action"
                        )

                    next_state, reward, done = (
                        environment.step(
                            state,
                            action,
                        )
                    )

                    # An unchanged position means the action
                    # was invalid in the V1 environment.
                    if (
                        next_state.position
                        == state.position
                    ):
                        invalid_actions += 1

                    total_reward += reward
                    steps += 1

                    state = next_state

                # --------------------------------------------------
                # Episode metrics
                # --------------------------------------------------

                cleaned = (
                    initial_dirty
                    - len(state.dirty_cells)
                )

                rewards.append(total_reward)
                steps_list.append(steps)
                cleaned_list.append(cleaned)
                completed_list.append(done)
                invalid_list.append(invalid_actions)

            finally:

                # Restore epsilon for factory-created
                # Q-Learning agents.
                if episode_original_epsilon is not None:
                    episode_agent.epsilon = (
                        episode_original_epsilon
                    )

    finally:

        # Restore epsilon for an externally supplied
        # Q-Learning agent.
        if original_epsilon is not None:
            agent.epsilon = original_epsilon

    # --------------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------------

    return EvaluationResult(
        episodes=episodes,
        average_reward=mean(rewards),
        average_steps=mean(steps_list),
        average_cleaned_cells=mean(cleaned_list),
        completion_rate=mean(completed_list),
        average_invalid_actions=mean(invalid_list),
    )