"""
Turn a validated :class:`ScenarioConfig` into runnable objects.

This is the only place that knows how config values map onto the
environment, agents, trainer and simulation.
"""

from __future__ import annotations

from typing import Callable

from cleaning_vehicle.agents.base_agent import BaseAgent
from cleaning_vehicle.agents.greedy_agent import GreedyAgent
from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.agents.random_agent import RandomAgent
from cleaning_vehicle.config.schema import ConfigError, ScenarioConfig
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.rewards import RewardCalculator
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.training.evaluator import (
    EvaluationResult,
    evaluate_agent,
)
from cleaning_vehicle.training.trainer import (
    ProgressCallback,
    TrainingResult,
    train_q_learning,
)
from cleaning_vehicle.visualization.simulation import Simulation


def build_environment(config: ScenarioConfig) -> CleaningEnvironment:
    """A fresh environment (its dynamics are stateful)."""

    env = config.environment

    return CleaningEnvironment(
        rows=env.grid.rows,
        cols=env.grid.cols,
        obstacles=frozenset(env.static_obstacles),
        charging_station=env.charging_station,
        max_battery=env.battery.max,
        low_battery_threshold=env.battery.low_threshold,
        reward_calculator=RewardCalculator(env.rewards),
        dynamic_obstacles=frozenset(env.dynamic_obstacles.positions),
        dynamic_obstacle_move_probability=(
            env.dynamic_obstacles.move_probability
        ),
        dirt_spawn_probability=env.dirt.spawn_probability,
        max_dirty_cells=env.dirt.max_dirty_cells,
        observation_radius=env.observation.radius,
        seed=config.seed,
    )


def build_state_factory(config: ScenarioConfig) -> Callable[[], State]:
    """Factory returning the configured initial State."""

    env = config.environment

    def factory() -> State:
        return State(
            position=env.start_position,
            dirty_cells=frozenset(env.dirt.initial_cells),
            battery=env.battery.initial,
        )

    return factory


def build_agent(config: ScenarioConfig) -> BaseAgent:
    """Create the (untrained) agent named in the config."""

    algorithm = config.agent.algorithm
    params = dict(config.agent.hyperparameters)

    if algorithm == "random":
        return RandomAgent(seed=config.seed)

    if algorithm == "greedy":
        return GreedyAgent()

    if algorithm == "q_learning":
        return QLearningAgent(seed=config.seed, **params)

    raise ConfigError(f"No agent implementation for '{algorithm}'")


def train_agent(
    config: ScenarioConfig,
    agent: BaseAgent,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult | None:
    """Train ``agent`` if its config says it needs training."""

    if not config.agent.requires_training:
        return None

    if not isinstance(agent, QLearningAgent):
        raise ConfigError(
            f"No trainer available for '{config.agent.algorithm}'"
        )

    return train_q_learning(
        environment=build_environment(config),
        agent=agent,
        initial_state_factory=build_state_factory(config),
        episodes=config.training.episodes,
        max_steps=config.training.max_steps,
        use_observations=config.partial_observation,
        progress_callback=progress_callback,
    )


def prepare_agent(
    config: ScenarioConfig,
    progress_callback: ProgressCallback | None = None,
) -> BaseAgent:
    """Build the agent and train it when needed. Ready to run."""

    agent = build_agent(config)
    train_agent(config, agent, progress_callback)

    if isinstance(agent, QLearningAgent):
        agent.epsilon = 0.0   # act greedily when demonstrating

    return agent


def build_simulation(
    config: ScenarioConfig,
    agent: BaseAgent,
) -> Simulation:
    """A simulation of ``agent`` in a fresh environment."""

    return Simulation(
        environment_factory=lambda: build_environment(config),
        state_factory=build_state_factory(config),
        agent=agent,
        partial_observation=config.partial_observation,
        max_steps=config.training.max_steps,
    )


def evaluate(config: ScenarioConfig, agent: BaseAgent) -> EvaluationResult:
    """Evaluate ``agent`` (no learning) using the config's settings."""

    return evaluate_agent(
        environment=build_environment(config),
        agent=agent,
        initial_state_factory=build_state_factory(config),
        episodes=config.evaluation.episodes,
        max_steps=config.training.max_steps,
        use_observations=config.partial_observation,
    )
