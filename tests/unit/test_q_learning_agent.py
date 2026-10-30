from cleaning_vehicle.environment.observation import Observation
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


# ==========================================================
# V2 - Partial Observation
# ==========================================================


def test_q_learning_agent_select_observation_returns_action():
    """
    V2:
    QLearningAgent must be able to select an action from
    a partial Observation.
    """

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (2, 3),
        }),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert isinstance(action, Action)


def test_q_learning_agent_select_observation_uses_observation_as_q_state():
    """
    V2:
    Q-values for an Observation must be independent from
    Q-values for a different Observation.
    """

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    observation_one = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (2, 3),
        }),
        dynamic_obstacles=frozenset(),
    )

    observation_two = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (0, 0),
        }),
        dynamic_obstacles=frozenset(),
    )

    agent.q_table[
        (observation_one, Action.RIGHT)
    ] = 10.0

    agent.q_table[
        (observation_two, Action.LEFT)
    ] = 20.0

    assert (
        agent.best_observation_action(observation_one)
        == Action.RIGHT
    )

    assert (
        agent.best_observation_action(observation_two)
        == Action.LEFT
    )


def test_q_learning_agent_select_observation_is_greedy_when_epsilon_zero():
    """
    V2:
    With epsilon=0, the agent should select the action
    with the highest Q-value for the observation.
    """

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (2, 3),
        }),
        dynamic_obstacles=frozenset(),
    )

    agent.q_table[
        (observation, Action.RIGHT)
    ] = 10.0

    agent.q_table[
        (observation, Action.LEFT)
    ] = 1.0

    agent.q_table[
        (observation, Action.UP)
    ] = 2.0

    agent.q_table[
        (observation, Action.DOWN)
    ] = 3.0

    assert (
        agent.select_observation(observation)
        == Action.RIGHT
    )


def test_q_learning_agent_select_observation_explores_when_epsilon_one():
    """
    V2:
    With epsilon=1, the agent should use random exploration.
    """

    agent = QLearningAgent(
        epsilon=1.0,
        seed=42,
    )

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    actions = {
        agent.select_observation(observation)
        for _ in range(20)
    }

    assert actions
    assert actions.issubset(set(Action))


def test_q_learning_agent_observation_q_values_are_independent():
    """
    V2:
    Two observations representing different partial views
    must have independent Q-values.
    """

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    observation_one = Observation(
        position=(1, 1),
        battery=10,
        dirty_cells=frozenset({
            (1, 2),
        }),
        dynamic_obstacles=frozenset(),
    )

    observation_two = Observation(
        position=(1, 1),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    agent.q_table[
        (observation_one, Action.RIGHT)
    ] = 15.0

    assert (
        agent.get_observation_q_value(
            observation_one,
            Action.RIGHT,
        )
        == 15.0
    )

    assert (
        agent.get_observation_q_value(
            observation_two,
            Action.RIGHT,
        )
        == 0.0
    )


def test_q_learning_agent_observation_does_not_modify_v1_q_table():
    """
    V2:
    Adding observation-based Q-values must not interfere
    with the existing State-based V1 Q-table.
    """

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    state = State(
        position=(1, 1),
        battery=10,
        dirty_cells=frozenset({
            (1, 2),
        }),
    )

    observation = Observation(
        position=(1, 1),
        battery=10,
        dirty_cells=frozenset({
            (1, 2),
        }),
        dynamic_obstacles=frozenset(),
    )

    agent.q_table[
        (state, Action.RIGHT)
    ] = 5.0

    agent.q_table[
        (observation, Action.RIGHT)
    ] = 20.0

    assert (
        agent.get_q_value(
            state,
            Action.RIGHT,
        )
        == 5.0
    )

    assert (
        agent.get_observation_q_value(
            observation,
            Action.RIGHT,
        )
        == 20.0
    )