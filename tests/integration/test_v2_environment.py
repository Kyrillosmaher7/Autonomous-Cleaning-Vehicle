from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


def test_v2_observation_contains_nearby_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
        dynamic_obstacles=frozenset({(2, 3)}),
    )

    state = State(
        position=(2, 2),
        battery=20,
        dirty_cells=frozenset(),
    )

    observation = environment.get_observation(state)

    assert (2, 3) in observation.dynamic_obstacles


def test_v2_observation_hides_distant_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
        dynamic_obstacles=frozenset({(4, 4)}),
    )

    state = State(
        position=(2, 2),
        battery=20,
        dirty_cells=frozenset(),
    )

    observation = environment.get_observation(state)

    assert (4, 4) not in observation.dynamic_obstacles


def test_v2_nearby_dirt_is_visible():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
    )

    state = State(
        position=(2, 2),
        battery=20,
        dirty_cells=frozenset({
            (2, 3),
        }),
    )

    observation = environment.get_observation(state)

    assert (2, 3) in observation.dirty_cells


def test_v2_distant_dirt_is_hidden():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
    )

    state = State(
        position=(2, 2),
        battery=20,
        dirty_cells=frozenset({
            (4, 4),
        }),
    )

    observation = environment.get_observation(state)

    assert (4, 4) not in observation.dirty_cells


def test_v2_battery_decreases_after_movement():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
    )

    state = State(
        position=(1, 1),
        battery=10,
        dirty_cells=frozenset(),
    )

    next_state, _, _ = environment.step(
        state,
        Action.RIGHT,
    )

    assert next_state.battery < state.battery


def test_v2_robot_recharges_at_charging_station():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        charging_station=(0, 0),
        max_battery=20,
    )

    state = State(
        position=(0, 1),
        battery=5,
        dirty_cells=frozenset(),
    )

    next_state, _, _ = environment.step(
        state,
        Action.LEFT,
    )

    assert next_state.position == (0, 0)
    assert next_state.battery == 20