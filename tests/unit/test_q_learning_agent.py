import pytest

from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.state import State


@pytest.fixture
def state():
    return State(
        position=(2, 2),
        dirty_cells=frozenset({(2, 4)}),
        battery=20,
    )


def test_initial_q_values_are_zero(state):
    agent = QLearningAgent()

    for action in Action:
        assert agent.get_q_value(state, action) == 0.0


def test_q_value_update(state):
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    next_state = State(
        position=(2, 3),
        dirty_cells=state.dirty_cells,
        battery=19,
    )

    agent.update(
        state=state,
        action=Action.RIGHT,
        reward=10.0,
        next_state=next_state,
        done=False,
    )

    assert agent.get_q_value(
        state,
        Action.RIGHT,
    ) == pytest.approx(5.0)


def test_terminal_update_does_not_bootstrap(state):
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    next_state = State(
        position=(2, 3),
        dirty_cells=frozenset(),
        battery=19,
    )

    agent.update(
        state=state,
        action=Action.RIGHT,
        reward=20.0,
        next_state=next_state,
        done=True,
    )

    assert agent.get_q_value(
        state,
        Action.RIGHT,
    ) == pytest.approx(10.0)


def test_best_action_selects_highest_q_value(state):
    agent = QLearningAgent()

    agent.q_table[(state, Action.UP)] = 1.0
    agent.q_table[(state, Action.RIGHT)] = 5.0
    agent.q_table[(state, Action.DOWN)] = 2.0
    agent.q_table[(state, Action.LEFT)] = 3.0

    assert agent.best_action(state) == Action.RIGHT


def test_epsilon_one_explores(state):
    agent = QLearningAgent(
        epsilon=1.0,
        seed=42,
    )

    actions = {
        agent.select_action(state)
        for _ in range(100)
    }

    assert len(actions) > 1


def test_epsilon_zero_exploits(state):
    agent = QLearningAgent(
        epsilon=0.0,
    )

    agent.q_table[(state, Action.RIGHT)] = 10.0

    for _ in range(20):
        assert agent.select_action(state) == Action.RIGHT


def test_epsilon_decay():
    agent = QLearningAgent(
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.1,
    )

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.5)

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.25)

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.125)

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.1)


def test_q_table_size(state):
    agent = QLearningAgent()

    assert agent.table_size == 0

    agent.get_q_value(
        state,
        Action.RIGHT,
    )

    assert agent.table_size == 1