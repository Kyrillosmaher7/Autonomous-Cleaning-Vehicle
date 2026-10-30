from cleaning_vehicle.environment.observation import Observation


def test_observation_stores_robot_position():
    observation = Observation(
        position=(2, 3),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    assert observation.position == (2, 3)


def test_observation_stores_battery():
    observation = Observation(
        position=(2, 3),
        battery=7,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    assert observation.battery == 7


def test_observation_stores_visible_dirty_cells():
    dirty_cells = frozenset({
        (1, 2),
        (2, 3),
    })

    observation = Observation(
        position=(2, 3),
        battery=10,
        dirty_cells=dirty_cells,
        dynamic_obstacles=frozenset(),
    )

    assert observation.dirty_cells == dirty_cells


def test_observation_stores_visible_dynamic_obstacles():
    obstacles = frozenset({
        (1, 3),
        (3, 3),
    })

    observation = Observation(
        position=(2, 3),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=obstacles,
    )

    assert observation.dynamic_obstacles == obstacles


def test_observation_is_immutable():
    observation = Observation(
        position=(2, 3),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    try:
        observation.battery = 5
        assert False
    except AttributeError:
        pass


def test_observation_is_hashable():
    observation = Observation(
        position=(2, 3),
        battery=10,
        dirty_cells=frozenset(),
        dynamic_obstacles=frozenset(),
    )

    assert isinstance(hash(observation), int)


def test_observation_as_tuple_contains_all_information():
    observation = Observation(
        position=(2, 3),
        battery=10,
        dirty_cells=frozenset({(2, 4)}),
        dynamic_obstacles=frozenset({(1, 3)}),
    )

    assert observation.as_tuple() == (
        (2, 3),
        10,
        frozenset({(2, 4)}),
        frozenset({(1, 3)}),
    )