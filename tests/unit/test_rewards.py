from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.rewards import RewardCalculator, RewardConfig
from cleaning_vehicle.environment.state import State


def make_state(
    position=(2, 2),
    dirty_cells=frozenset(),
    battery=20,
):
    return State(
        position=position,
        dirty_cells=dirty_cells,
        battery=battery,
    )


def test_default_reward_configuration():
    config = RewardConfig()

    assert config.clean == 10.0
    assert config.move == -1.0
    assert config.invalid == -2.0
    assert config.completion == 20.0


def test_reward_calculator():
    calculator = RewardCalculator()

    assert calculator.clean_reward() == 10.0
    assert calculator.movement_reward() == -1.0
    assert calculator.invalid_reward() == -2.0
    assert calculator.completion_reward() == 20.0


def test_custom_reward_configuration():
    config = RewardConfig(
        clean=15.0,
        move=-0.5,
        invalid=-3.0,
        completion=25.0,
    )

    calculator = RewardCalculator(config)

    assert calculator.clean_reward() == 15.0
    assert calculator.movement_reward() == -0.5
    assert calculator.invalid_reward() == -3.0
    assert calculator.completion_reward() == 25.0


def test_valid_move_returns_movement_penalty():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        dirty_cells=frozenset({(5, 5)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (2, 3)
    assert next_state.battery == 19
    assert reward == -1.0
    assert done is False

def test_cleaning_dirty_cell_gives_positive_reward():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3), (8, 8)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (2, 3)
    assert next_state.dirty_cells == frozenset({(8, 8)})

    # movement -1 + cleaning +10
    assert reward == 9.0
    assert done is False


def test_cleaning_last_dirty_cell_finishes_episode():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (2, 3)
    assert next_state.dirty_cells == frozenset()

    # movement -1
    # cleaning +10
    # completion +20
    assert reward == 29.0
    assert done is True


def test_boundary_action_is_penalized():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(0, 0),
        dirty_cells=frozenset({(8, 8)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.UP,
    )

    assert next_state.position == (0, 0)
    assert reward == -2.0
    assert done is False


def test_left_boundary_action_is_penalized():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(5, 0),
        dirty_cells=frozenset({(8, 8)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.LEFT,
    )

    assert next_state.position == (5, 0)
    assert reward == -2.0
    assert done is False


def test_obstacle_action_is_penalized():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
        obstacles=frozenset({(2, 3)}),
    )

    state = make_state(
        position=(2, 2),
        dirty_cells=frozenset({(8, 8)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (2, 2)
    assert reward == -2.0
    assert done is False


def test_obstacle_does_not_remove_dirt():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
        obstacles=frozenset({(2, 3)}),
    )

    dirty = frozenset({(2, 3), (8, 8)})

    state = make_state(
        position=(2, 2),
        dirty_cells=dirty,
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert next_state.dirty_cells == dirty


def test_normal_move_does_not_remove_dirt():
    dirty = frozenset({(8, 8)})

    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(2, 2),
        dirty_cells=dirty,
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert next_state.dirty_cells == dirty


def test_is_done_when_no_dirty_cells():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        dirty_cells=frozenset(),
    )

    assert env.is_done(state) is True


def test_is_not_done_when_dirty_cells_remain():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        dirty_cells=frozenset({(5, 5)}),
    )

    assert env.is_done(state) is False


def test_invalid_action_does_not_change_state():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(0, 0),
        dirty_cells=frozenset({(5, 5)}),
    )

    next_state, _, _ = env.step(
        state,
        Action.UP,
    )

    assert next_state == state


def test_cleaning_is_immutable():
    dirty = frozenset({(2, 3), (8, 8)})

    env = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    state = make_state(
        position=(2, 2),
        dirty_cells=dirty,
    )

    next_state, _, _ = env.step(
        state,
        Action.RIGHT,
    )

    assert state.dirty_cells == dirty
    assert next_state.dirty_cells == frozenset({(8, 8)})


def test_custom_rewards_are_used():
    calculator = RewardCalculator(
        RewardConfig(
            clean=100.0,
            move=-0.5,
            invalid=-10.0,
            completion=50.0,
        )
    )

    env = CleaningEnvironment(
        rows=10,
        cols=10,
        reward_calculator=calculator,
    )

    state = make_state(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3), (8, 8)}),
    )

    next_state, reward, done = env.step(
        state,
        Action.RIGHT,
    )

    assert reward == 99.5
    assert done is False