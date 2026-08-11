import pytest

from cleaning_vehicle.agents import BaseAgent, RandomAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.state import State


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent()


def test_random_agent_is_base_agent():
    agent = RandomAgent(seed=42)

    assert isinstance(agent, BaseAgent)


def test_random_agent_returns_valid_action():
    agent = RandomAgent(seed=42)

    state = State(
        position=(5, 5),
        dirty_cells=frozenset({(1, 1)}),
    )

    action = agent.select_action(state)

    assert isinstance(action, Action)


def test_random_agent_reproducible_with_seed():
    state = State(
        position=(5, 5),
        dirty_cells=frozenset({(1, 1)}),
    )

    agent_a = RandomAgent(seed=42)
    agent_b = RandomAgent(seed=42)

    actions_a = [
        agent_a.select_action(state)
        for _ in range(20)
    ]

    actions_b = [
        agent_b.select_action(state)
        for _ in range(20)
    ]

    assert actions_a == actions_b


def test_random_agent_can_select_multiple_actions():
    agent = RandomAgent(seed=42)

    state = State(
        position=(5, 5),
        dirty_cells=frozenset({(1, 1)}),
    )

    actions = {
        agent.select_action(state)
        for _ in range(100)
    }

    assert len(actions) > 1