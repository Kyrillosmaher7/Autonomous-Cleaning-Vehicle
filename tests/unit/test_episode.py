from cleaning_vehicle.agents import BaseAgent
from cleaning_vehicle.utils.episode import EpisodeResult, run_episode
from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


class RightAgent(BaseAgent):

    def select_action(self, state: State) -> Action:
        return Action.RIGHT


class UpAgent(BaseAgent):

    def select_action(self, state: State) -> Action:
        return Action.UP


def test_episode_result():
    result = EpisodeResult(
        total_reward=10.0,
        steps=5,
        cleaned_cells=1,
        completed=True,
        invalid_actions=2,
    )

    assert result.total_reward == 10.0
    assert result.steps == 5
    assert result.cleaned_cells == 1
    assert result.completed is True
    assert result.invalid_actions == 2


def test_episode_completes_when_dirt_is_reached():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({(0, 1)}),
    )

    result = run_episode(
        environment=environment,
        agent=RightAgent(),
        initial_state=state,
        max_steps=10,
    )

    assert result.completed is True
    assert result.steps == 1
    assert result.cleaned_cells == 1

    # -1 movement + 10 cleaning + 20 completion
    assert result.total_reward == 29.0
    assert result.invalid_actions == 0


def test_episode_stops_at_max_steps():
    environment = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({(9, 9)}),
    )

    result = run_episode(
        environment=environment,
        agent=RightAgent(),
        initial_state=state,
        max_steps=3,
    )

    assert result.completed is False
    assert result.steps == 3
    assert result.cleaned_cells == 0


def test_invalid_actions_are_counted():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({(2, 2)}),
    )

    result = run_episode(
        environment=environment,
        agent=UpAgent(),
        initial_state=state,
        max_steps=5,
    )

    assert result.completed is False
    assert result.steps == 5
    assert result.cleaned_cells == 0
    assert result.invalid_actions == 5
    assert result.total_reward == -10.0


def test_cleaned_cells_are_counted():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({
            (0, 1),
            (0, 2),
        }),
    )

    result = run_episode(
        environment=environment,
        agent=RightAgent(),
        initial_state=state,
        max_steps=10,
    )

    assert result.completed is True
    assert result.steps == 2
    assert result.cleaned_cells == 2

    # First cell: -1 + 10
    # Second cell: -1 + 10 + 20
    assert result.total_reward == 38.0