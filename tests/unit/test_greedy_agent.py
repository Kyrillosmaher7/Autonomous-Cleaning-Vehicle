from cleaning_vehicle.agents import BaseAgent, GreedyAgent
from cleaning_vehicle.environment.actions import Action
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