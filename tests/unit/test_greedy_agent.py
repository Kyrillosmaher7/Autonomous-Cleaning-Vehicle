from cleaning_vehicle.agents import BaseAgent, GreedyAgent
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.observation import Observation
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.agents import GreedyAgent
from cleaning_vehicle.utils.episode import run_episode
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State

def test_greedy_agent_is_base_agent():
    agent = GreedyAgent()

    assert isinstance(agent, BaseAgent)


def test_greedy_moves_down_when_target_is_below():
    agent = GreedyAgent()

    state = State(
        position=(1, 1),
        dirty_cells=frozenset({(3, 1)}),
    )

    assert agent.select_action(state) == Action.DOWN


def test_greedy_moves_up_when_target_is_above():
    agent = GreedyAgent()

    state = State(
        position=(3, 1),
        dirty_cells=frozenset({(1, 1)}),
    )

    assert agent.select_action(state) == Action.UP


def test_greedy_moves_right_when_target_is_right():
    agent = GreedyAgent()

    state = State(
        position=(1, 1),
        dirty_cells=frozenset({(1, 4)}),
    )

    assert agent.select_action(state) == Action.RIGHT


def test_greedy_moves_left_when_target_is_left():
    agent = GreedyAgent()

    state = State(
        position=(1, 4),
        dirty_cells=frozenset({(1, 1)}),
    )

    assert agent.select_action(state) == Action.LEFT


def test_greedy_selects_nearest_dirty_cell():
    agent = GreedyAgent()

    state = State(
        position=(5, 5),
        dirty_cells=frozenset({
            (5, 8),  # distance = 3
            (1, 5),  # distance = 4
        }),
    )

    assert agent.select_action(state) == Action.RIGHT


def test_greedy_handles_diagonal_target():
    agent = GreedyAgent()

    state = State(
        position=(1, 1),
        dirty_cells=frozenset({(3, 4)}),
    )

    # Vertical and horizontal movement both reduce
    # Manhattan distance.
    # Priority chooses DOWN before RIGHT.
    assert agent.select_action(state) == Action.DOWN


def test_greedy_returns_deterministic_action_without_dirt():
    agent = GreedyAgent()

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
    )

    assert agent.select_action(state) == Action.UP


def test_manhattan_distance():
    assert (
        GreedyAgent._manhattan_distance(
            (2, 2),
            (5, 6),
        )
        == 7
    )
def test_greedy_agent_can_clean_episode():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({
            (0, 2),
            (2, 2),
        }),
    )

    result = run_episode(
        environment=environment,
        agent=GreedyAgent(),
        initial_state=state,
        max_steps=20,
    )

    assert result.completed is True
    assert result.cleaned_cells == 2
    assert result.steps <= 20


# ==========================================================
# V2 - Partial Observation
# ==========================================================


def test_greedy_agent_select_observation_moves_toward_visible_dirty_cell():
    """
    V2:
    GreedyAgent should move toward the nearest visible dirty cell.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (1, 2),
        }),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert action == Action.UP


def test_greedy_agent_select_observation_moves_right_toward_visible_dirty_cell():
    """
    V2:
    GreedyAgent should move toward a visible dirty cell to the right.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (2, 3),
        }),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert action == Action.RIGHT


def test_greedy_agent_select_observation_moves_left_toward_visible_dirty_cell():
    """
    V2:
    GreedyAgent should move toward a visible dirty cell to the left.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (2, 1),
        }),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert action == Action.LEFT


def test_greedy_agent_select_observation_moves_down_toward_visible_dirty_cell():
    """
    V2:
    GreedyAgent should move toward a visible dirty cell below.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (3, 2),
        }),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert action == Action.DOWN


def test_greedy_agent_select_observation_uses_nearest_visible_dirty_cell():
    """
    V2:
    When multiple dirty cells are visible, the nearest one
    should determine the selected action.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset({
            (4, 2),
            (2, 3),
            (0, 2),
        }),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert action == Action.RIGHT


def test_greedy_agent_select_observation_ignores_hidden_dirty_cells():
    """
    V2:
    The agent only receives visible dirty cells through the
    Observation.

    A distant dirty cell must therefore have no influence
    on the selected action.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    action = agent.select_observation(observation)

    assert action == Action.UP


def test_greedy_agent_select_observation_returns_fallback_when_no_visible_dirt():
    """
    V2:
    If no dirty cell is visible, the agent should return its
    deterministic fallback action.
    """

    agent = GreedyAgent()

    observation = Observation(
        position=(2, 2),
        battery=3,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset({
            (2, 3),
        }),
    )

    action = agent.select_observation(observation)

    assert action == Action.UP