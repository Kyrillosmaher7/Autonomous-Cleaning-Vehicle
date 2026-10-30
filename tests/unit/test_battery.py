from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import (
    CleaningEnvironment,
)
from cleaning_vehicle.environment.state import State


def test_valid_movement_consumes_one_battery():
    env = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=10,
        charging_station=(4, 4),
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=10,
    )

    next_state = env.move(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (2, 3)
    assert next_state.battery == 9


def test_invalid_movement_does_not_consume_battery():
    env = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=10,
        charging_station=(4, 4),
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({(2, 2)}),
        battery=10,
    )

    next_state = env.move(
        state,
        Action.UP,
    )

    assert next_state.position == (0, 0)
    assert next_state.battery == 10


def test_reaching_charging_station_restores_battery():
    env = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=20,
        charging_station=(2, 2),
    )

    state = State(
        position=(2, 3),
        dirty_cells=frozenset({(4, 4)}),
        battery=5,
    )

    next_state = env.move(
        state,
        Action.LEFT,
    )

    assert next_state.position == (2, 2)
    assert next_state.battery == 20


def test_charging_station_cannot_be_obstacle():
    try:
        CleaningEnvironment(
            rows=5,
            cols=5,
            obstacles=frozenset({(2, 2)}),
            charging_station=(2, 2),
        )
    except ValueError:
        return

    assert False, (
        "Expected ValueError when charging station "
        "is an obstacle."
    )


def test_charging_station_must_be_inside_grid():
    try:
        CleaningEnvironment(
            rows=5,
            cols=5,
            charging_station=(5, 5),
        )
    except ValueError:
        return

    assert False, (
        "Expected ValueError for invalid charging station."
    )


def test_battery_zero_ends_episode_away_from_charger():
    env = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(0, 0),
        max_battery=10,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(4, 4)}),
        battery=0,
    )

    assert env.is_done(state) is True


def test_battery_zero_at_charger_does_not_end_due_to_battery():
    env = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(2, 2),
        max_battery=10,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(4, 4)}),
        battery=0,
    )

    assert env.is_done(state) is False


def test_state_battery_is_part_of_q_learning_state():
    state_a = State(
        position=(1, 1),
        dirty_cells=frozenset({(3, 3)}),
        battery=10,
    )

    state_b = State(
        position=(1, 1),
        dirty_cells=frozenset({(3, 3)}),
        battery=5,
    )

    assert state_a != state_b
    assert state_a.as_tuple() != state_b.as_tuple()


def test_state_recharge():
    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=3,
    )

    charged = state.recharge(20)

    assert charged.position == state.position
    assert charged.dirty_cells == state.dirty_cells
    assert charged.battery == 20


def test_state_consume_battery():
    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=5,
    )

    next_state = state.consume_battery()

    assert next_state.battery == 4