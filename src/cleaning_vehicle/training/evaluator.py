from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterator

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State

AgentFactory = Callable[[], BaseAgent]
StateFactory = Callable[[], State]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregated metrics from evaluation episodes."""

    episodes: int
    average_reward: float
    average_steps: float
    average_cleaned_cells: float
    completion_rate: float
    average_invalid_actions: float


@contextmanager
def _greedy(agent: BaseAgent) -> Iterator[None]:
    """Temporarily disable exploration for Q-Learning agents."""

    if not isinstance(agent, QLearningAgent):
        yield
        return

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        yield
    finally:
        agent.epsilon = original_epsilon


def _select_action(
    environment: CleaningEnvironment,
    agent: BaseAgent,
    state: State,
    use_observations: bool,
) -> Action:
    """Ask the agent for an action using the configured input style."""

    if use_observations:
        action = agent.select_observation(
            environment.get_observation(state)
        )
    else:
        action = agent.select_action(state)

    if not isinstance(action, Action):
        raise TypeError(
            "agent action-selection method must return an Action"
        )

    return action


def _run_episode(
    environment: CleaningEnvironment,
    agent: BaseAgent,
    initial_state: State,
    max_steps: int,
    use_observations: bool,
) -> tuple[float, int, int, bool, int]:
    """
    Run one evaluation episode.

    Returns (total_reward, steps, cleaned_cells, done, invalid_actions).
    """

    state = initial_state
    initial_dirty = len(state.dirty_cells)

    total_reward = 0.0
    steps = 0
    invalid_actions = 0
    done = len(state.dirty_cells) == 0

    while not done and steps < max_steps:

        action = _select_action(
            environment, agent, state, use_observations
        )

        next_state, reward, done = environment.step(state, action)

        # An unchanged position means the move was invalid.
        if next_state.position == state.position:
            invalid_actions += 1

        total_reward += reward
        steps += 1
        state = next_state

    cleaned = initial_dirty - len(state.dirty_cells)

    return total_reward, steps, cleaned, done, invalid_actions


def evaluate_agent(
    environment: CleaningEnvironment,
    agent: BaseAgent | None = None,
    agent_factory: AgentFactory | None = None,
    initial_state_factory: StateFactory | None = None,
    episodes: int = 100,
    max_steps: int = 100,
    use_observations: bool = False,
) -> EvaluationResult:
    """
    Evaluate an agent without learning.

    Provide either ``agent`` (one agent reused for every episode) or
    ``agent_factory`` (a fresh agent per episode).

    Q-Learning agents are evaluated greedily: epsilon is set to zero
    for the evaluation and restored afterwards.

    ``use_observations=False`` feeds the agent the full State (V1);
    ``True`` feeds it partial Observations (V2).
    """

    if agent is None and agent_factory is None:
        raise ValueError("Either agent or agent_factory must be provided")

    if agent is not None and agent_factory is not None:
        raise ValueError("Provide either agent or agent_factory, not both")

    if initial_state_factory is None:
        raise ValueError("initial_state_factory must be provided")

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    rewards: list[float] = []
    steps_list: list[int] = []
    cleaned_list: list[int] = []
    completed_list: list[bool] = []
    invalid_list: list[int] = []

    for _ in range(episodes):

        episode_agent = agent_factory() if agent_factory else agent

        with _greedy(episode_agent):
            reward, steps, cleaned, done, invalid = _run_episode(
                environment,
                episode_agent,
                initial_state_factory(),
                max_steps,
                use_observations,
            )

        rewards.append(reward)
        steps_list.append(steps)
        cleaned_list.append(cleaned)
        completed_list.append(done)
        invalid_list.append(invalid)

    return EvaluationResult(
        episodes=episodes,
        average_reward=mean(rewards),
        average_steps=mean(steps_list),
        average_cleaned_cells=mean(cleaned_list),
        completion_rate=mean(completed_list),
        average_invalid_actions=mean(invalid_list),
    )
