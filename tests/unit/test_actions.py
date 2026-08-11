from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


def test_action_deltas():
    assert Action.UP.delta == (-1, 0)
    assert Action.DOWN.delta == (1, 0)
    assert Action.LEFT.delta == (0, -1)
    assert Action.RIGHT.delta == (0, 1)


def test_action_space_contains_four_actions():
    assert len(Action) == 4


def test_move_up():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.UP)

    assert new_state.position == (4, 5)


def test_move_down():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.DOWN)

    assert new_state.position == (6, 5)


def test_move_left():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.LEFT)

    assert new_state.position == (5, 4)


def test_move_right():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.RIGHT)

    assert new_state.position == (5, 6)


def test_top_boundary_blocks_up():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(0, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.UP)

    assert new_state.position == (0, 5)


def test_bottom_boundary_blocks_down():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(9, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.DOWN)

    assert new_state.position == (9, 5)


def test_left_boundary_blocks_left():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 0),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.LEFT)

    assert new_state.position == (5, 0)


def test_right_boundary_blocks_right():
    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 9),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.RIGHT)

    assert new_state.position == (5, 9)


def test_obstacle_blocks_movement():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
        obstacles=frozenset({(5, 6)}),
    )

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.RIGHT)

    assert new_state.position == (5, 5)


def test_movement_around_obstacle():
    env = CleaningEnvironment(
        rows=10,
        cols=10,
        obstacles=frozenset({(5, 6)}),
    )

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
    )

    new_state = env.move(state, Action.UP)

    assert new_state.position == (4, 5)


def test_movement_preserves_dirty_cells():
    dirty = frozenset({(2, 2), (7, 8)})

    env = CleaningEnvironment(rows=10, cols=10)

    state = State(
        position=(5, 5),
        dirty_cells=dirty,
    )

    new_state = env.move(state, Action.RIGHT)

    assert new_state.position == (5, 6)
    assert new_state.dirty_cells == dirty