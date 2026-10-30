import pytest

from cleaning_vehicle.agents.greedy_agent import GreedyAgent
from cleaning_vehicle.agents.random_agent import RandomAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.visualization.simulation import (
    END_BATTERY,
    END_COMPLETED,
    END_TIME_LIMIT,
    Simulation,
)


def _simulation(
    *,
    dirty=frozenset({(0, 2)}),
    battery=10,
    max_steps=50,
    partial=False,
    env_kwargs=None,
    agent=None,
):
    env_kwargs = env_kwargs or {}

    return Simulation(
        environment_factory=lambda: CleaningEnvironment(
            rows=3, cols=3, max_battery=10, **env_kwargs
        ),
        state_factory=lambda: State((0, 0), frozenset(dirty), battery),
        agent=agent or GreedyAgent(),
        partial_observation=partial,
        max_steps=max_steps,
    )


def test_greedy_completes_a_small_grid():
    sim = _simulation()

    while sim.step():
        pass

    assert sim.end_reason == END_COMPLETED
    assert sim.cleaned_count == 1
    assert sim.step_count == 2


def test_step_returns_none_after_finishing():
    sim = _simulation()

    while sim.step():
        pass

    assert sim.step() is None


def test_time_limit_ends_the_episode():
    sim = _simulation(max_steps=1)

    sim.step()

    assert sim.end_reason == END_TIME_LIMIT
    assert sim.step_count == 1


def test_battery_depletion_is_reported():
    sim = _simulation(
        dirty=frozenset({(2, 2)}),
        battery=1,
        agent=GreedyAgent(),
    )

    while sim.step():
        pass

    assert sim.end_reason == END_BATTERY


def test_observation_is_only_available_in_partial_mode():
    assert _simulation(partial=False).observation is None
    assert _simulation(partial=True).observation is not None


def test_partial_mode_feeds_the_agent_observations():
    sim = _simulation(partial=True)

    assert sim.step() is not None


def test_reset_restores_state_and_dynamic_obstacles():
    sim = _simulation(
        dirty=frozenset({(2, 2)}),
        max_steps=30,
        agent=RandomAgent(seed=1),
        env_kwargs={
            "dynamic_obstacles": frozenset({(1, 1)}),
            "dynamic_obstacle_move_probability": 1.0,
            "seed": 3,
        },
    )
    start_obstacles = set(sim.environment.dynamic_obstacles)

    moved = False

    for _ in range(10):
        sim.step()
        moved |= sim.environment.dynamic_obstacles != start_obstacles

    assert moved, "the obstacle should have moved during the run"

    sim.reset()

    assert sim.step_count == 0
    assert sim.total_reward == 0.0
    assert not sim.finished
    assert sim.state.position == (0, 0)
    assert sim.environment.dynamic_obstacles == start_obstacles


def test_invalid_max_steps_is_rejected():
    with pytest.raises(ValueError):
        _simulation(max_steps=0)
