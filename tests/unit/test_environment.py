from __future__ import annotations

import pytest

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State


# ==========================================================
# Basic Environment Tests
# ==========================================================


def test_environment_initializes_without_dynamic_obstacles():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
    )

    assert environment.rows == 5
    assert environment.cols == 5
    assert environment.dynamic_obstacles == frozenset()


def test_environment_initializes_with_dynamic_obstacles():
    dynamic_obstacles = frozenset({
        (2, 2),
        (4, 4),
    })

    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=dynamic_obstacles,
    )

    assert environment.dynamic_obstacles == dynamic_obstacles


# ==========================================================
# Dynamic Obstacle Validation
# ==========================================================


def test_dynamic_obstacle_must_be_inside_grid():
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            dynamic_obstacles=frozenset({
                (5, 5),
            }),
        )


def test_dynamic_obstacle_cannot_overlap_static_obstacle():
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            obstacles=frozenset({
                (2, 2),
            }),
            dynamic_obstacles=frozenset({
                (2, 2),
            }),
        )


def test_dynamic_obstacle_cannot_occupy_charging_station():
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            charging_station=(0, 0),
            dynamic_obstacles=frozenset({
                (0, 0),
            }),
        )


def test_dynamic_obstacle_can_be_at_valid_position():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
    )

    assert (2, 2) in environment.dynamic_obstacles


# ==========================================================
# Dynamic Obstacle Detection
# ==========================================================


def test_is_dynamic_obstacle_returns_true_for_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
    )

    assert environment._is_dynamic_obstacle((2, 2))


def test_is_dynamic_obstacle_returns_false_for_empty_cell():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
    )

    assert not environment._is_dynamic_obstacle((1, 1))


def test_dynamic_obstacle_is_not_traversable():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
    )

    assert not environment._is_valid_position((2, 2))


# ==========================================================
# Robot Movement and Dynamic Obstacles
# ==========================================================


def test_robot_cannot_move_into_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (1, 2),
        }),
    )

    state = State(
        position=(1, 1),
        dirty_cells=frozenset(),
        battery=20,
    )

    next_state = environment.move(
        state,
        Action.RIGHT,
    )

    assert next_state == state


def test_robot_can_move_to_cell_not_occupied_by_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (3, 3),
        }),
    )

    state = State(
        position=(1, 1),
        dirty_cells=frozenset(),
        battery=20,
    )

    next_state = environment.move(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (1, 2)
    assert next_state.battery == 19


def test_invalid_robot_move_into_dynamic_obstacle_gets_invalid_reward():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (1, 2),
        }),
    )

    state = State(
        position=(1, 1),
        dirty_cells=frozenset({
            (4, 4),
        }),
        battery=20,
    )

    next_state, reward, done = environment.step(
        state,
        Action.RIGHT,
    )

    assert next_state == state
    assert reward == -2.0
    assert not done


# ==========================================================
# Dynamic Obstacle Movement
# ==========================================================


def test_dynamic_obstacle_can_move():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    original_position = (2, 2)

    environment.update_dynamic_obstacles(
        robot_position=(0, 0),
    )

    assert environment.dynamic_obstacles
    assert environment.dynamic_obstacles != frozenset({
        original_position,
    })


def test_dynamic_obstacle_stays_when_move_probability_is_zero():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
        dynamic_obstacle_move_probability=0.0,
        seed=42,
    )

    original_obstacles = environment.dynamic_obstacles

    environment.update_dynamic_obstacles(
        robot_position=(0, 0),
    )

    assert environment.dynamic_obstacles == original_obstacles


def test_dynamic_obstacle_stays_inside_grid():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        charging_station=(2, 2),
        dynamic_obstacles=frozenset({
            (0, 0),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    environment.update_dynamic_obstacles(
        robot_position=(2, 2),
    )

    for position in environment.dynamic_obstacles:
        row, column = position

        assert 0 <= row < 3
        assert 0 <= column < 3


def test_dynamic_obstacle_does_not_move_into_static_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        obstacles=frozenset({
            (2, 3),
        }),
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    environment.update_dynamic_obstacles(
        robot_position=(0, 0),
    )

    assert (2, 3) not in environment.dynamic_obstacles


def test_dynamic_obstacle_does_not_move_into_robot():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 2),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    robot_position = (2, 3)

    environment.update_dynamic_obstacles(
        robot_position=robot_position,
    )

    assert robot_position not in environment.dynamic_obstacles


# ==========================================================
# Probability Validation
# ==========================================================


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, 2.0],
)
def test_dynamic_obstacle_probability_must_be_between_zero_and_one(
    probability: float,
):
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            dynamic_obstacle_move_probability=probability,
        )


def test_zero_probability_is_valid():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacle_move_probability=0.0,
    )

    assert (
        environment.dynamic_obstacle_move_probability
        == 0.0
    )


def test_one_probability_is_valid():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacle_move_probability=1.0,
    )

    assert (
        environment.dynamic_obstacle_move_probability
        == 1.0
    )


# ==========================================================
# Reproducibility
# ==========================================================


def test_dynamic_obstacle_movement_is_reproducible_with_same_seed():
    environment_one = CleaningEnvironment(
        rows=10,
        cols=10,
        dynamic_obstacles=frozenset({
            (5, 5),
            (7, 7),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    environment_two = CleaningEnvironment(
        rows=10,
        cols=10,
        dynamic_obstacles=frozenset({
            (5, 5),
            (7, 7),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    robot_position = (0, 0)

    environment_one.update_dynamic_obstacles(
        robot_position
    )

    environment_two.update_dynamic_obstacles(
        robot_position
    )

    assert (
        environment_one.dynamic_obstacles
        == environment_two.dynamic_obstacles
    )


# ==========================================================
# Step Integration
# ==========================================================


def test_step_updates_dynamic_obstacles():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (4, 4),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    state = State(
        position=(1, 1),
        dirty_cells=frozenset({
            (3, 3),
        }),
        battery=20,
    )

    original_obstacles = environment.dynamic_obstacles

    next_state, reward, done = environment.step(
        state,
        Action.RIGHT,
    )

    assert next_state.position == (1, 2)
    assert reward == -1.0
    assert not done

    assert (
        environment.dynamic_obstacles
        != original_obstacles
    )


def test_step_does_not_update_dynamic_obstacles_on_invalid_move():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (3, 3),
        }),
        dynamic_obstacle_move_probability=1.0,
        seed=42,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({
            (4, 4),
        }),
        battery=20,
    )

    original_obstacles = environment.dynamic_obstacles

    next_state, reward, done = environment.step(
        state,
        Action.UP,
    )

    assert next_state == state
    assert reward == -2.0
    assert not done

    assert (
        environment.dynamic_obstacles
        == original_obstacles
    )
    
    
# ==========================================================
# V2 Phase 2 — Dynamic Dirt
# ==========================================================


def test_dynamic_dirt_does_not_spawn_when_probability_is_zero():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dirt_spawn_probability=0.0,
        max_dirty_cells=5,
        seed=42,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert new_state == state
    assert new_state.dirty_cells == frozenset()


def test_dynamic_dirt_can_spawn_when_probability_is_one():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dirt_spawn_probability=1.0,
        max_dirty_cells=5,
        seed=42,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert len(new_state.dirty_cells) == 1


def test_dynamic_dirt_respects_maximum_dirty_cells():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dirt_spawn_probability=1.0,
        max_dirty_cells=2,
        seed=42,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({
            (0, 1),
            (0, 2),
        }),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert len(new_state.dirty_cells) == 2
    assert new_state.dirty_cells == state.dirty_cells


def test_dynamic_dirt_does_not_spawn_on_static_obstacle():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        obstacles=frozenset({
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
        }),
        charging_station=(2, 2),
        dirt_spawn_probability=1.0,
        max_dirty_cells=1,
        seed=42,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert new_state == state


def test_dynamic_dirt_does_not_spawn_on_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        charging_station=(2, 2),
        dynamic_obstacles=frozenset({
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
        }),
        dirt_spawn_probability=1.0,
        max_dirty_cells=1,
        seed=42,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert new_state == state


def test_dynamic_dirt_does_not_spawn_on_robot():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        charging_station=(2, 2),
        dirt_spawn_probability=1.0,
        max_dirty_cells=1,
        seed=42,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
        }),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert (0, 0) not in new_state.dirty_cells


def test_dynamic_dirt_does_not_spawn_on_charging_station():
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        charging_station=(2, 2),
        dirt_spawn_probability=1.0,
        max_dirty_cells=1,
        seed=42,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
        }),
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert (2, 2) not in new_state.dirty_cells


def test_dynamic_dirt_preserves_existing_dirty_cells():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dirt_spawn_probability=1.0,
        max_dirty_cells=5,
        seed=42,
    )

    existing_dirty = frozenset({
        (0, 1),
        (4, 4),
    })

    state = State(
        position=(2, 2),
        dirty_cells=existing_dirty,
        battery=20,
    )

    new_state = environment._spawn_dirt(state)

    assert existing_dirty.issubset(
        new_state.dirty_cells
    )


def test_dynamic_dirt_spawning_is_reproducible_with_same_seed():
    environment_one = CleaningEnvironment(
        rows=10,
        cols=10,
        dirt_spawn_probability=1.0,
        max_dirty_cells=5,
        seed=42,
    )

    environment_two = CleaningEnvironment(
        rows=10,
        cols=10,
        dirt_spawn_probability=1.0,
        max_dirty_cells=5,
        seed=42,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset(),
        battery=20,
    )

    state_one = environment_one._spawn_dirt(state)
    state_two = environment_two._spawn_dirt(state)

    assert state_one.dirty_cells == state_two.dirty_cells


def test_dynamic_dirt_is_integrated_into_step():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(4, 4),
        dirt_spawn_probability=1.0,
        max_dirty_cells=5,
        seed=42,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({
            (2, 3),
        }),
        battery=20,
    )

    next_state, reward, done = environment.step(
        state,
        Action.RIGHT,
    )

    assert (2, 3) not in next_state.dirty_cells

    # Dynamic dirt should have the opportunity to appear
    # after the existing dirt is cleaned.
    assert len(next_state.dirty_cells) >= 1

    assert reward >= 9.0


# ==========================================================
# V2 Phase 3 — Energy-Aware Battery Management
# ==========================================================


def test_low_battery_detection():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(0, 0),
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3)}),
        battery=5,
    )

    assert environment.is_low_battery(state) is True


def test_battery_above_low_threshold_is_not_low():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3)}),
        battery=6,
    )

    assert environment.is_low_battery(state) is False


def test_zero_battery_is_critical():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3)}),
        battery=0,
    )

    assert environment.is_critical_battery(state) is True


def test_battery_above_zero_is_not_critical():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3)}),
        battery=1,
    )

    assert environment.is_critical_battery(state) is False


def test_battery_is_healthy_above_low_threshold():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({(2, 3)}),
        battery=10,
    )

    assert environment.has_sufficient_battery(state) is True


def test_low_battery_requires_return_to_charger():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(0, 0),
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(3, 3),
        dirty_cells=frozenset({(4, 4)}),
        battery=5,
    )

    assert environment.should_return_to_charger(state) is True


def test_robot_at_charger_does_not_need_to_return():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(0, 0),
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset({(4, 4)}),
        battery=5,
    )

    assert environment.should_return_to_charger(state) is False


def test_healthy_battery_does_not_require_return():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        charging_station=(0, 0),
        max_battery=20,
        low_battery_threshold=5,
    )

    state = State(
        position=(3, 3),
        dirty_cells=frozenset({(4, 4)}),
        battery=10,
    )

    assert environment.should_return_to_charger(state) is False


def test_battery_threshold_must_be_valid():
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            max_battery=20,
            low_battery_threshold=0,
        )


def test_battery_threshold_cannot_exceed_max_battery():
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            max_battery=20,
            low_battery_threshold=21,
        )
def test_dynamic_dirt_does_not_end_episode_when_no_dirt():
    environment = CleaningEnvironment(
        rows=10,
        cols=10,
        charging_station=(0, 0),
        max_battery=50,
        dirt_spawn_probability=0.5,
        max_dirty_cells=5,
    )

    state = State(
        position=(0, 0),
        dirty_cells=frozenset(),
        battery=50,
    )

    assert environment.is_done(state) is False

# ==========================================================
# V2 Phase 4.2 — Partial Observation
# ==========================================================


def test_observation_radius_must_be_non_negative():
    with pytest.raises(ValueError):
        CleaningEnvironment(
            rows=5,
            cols=5,
            observation_radius=-1,
        )


def test_observation_contains_robot_position():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=15,
    )

    observation = environment.get_observation(state)

    assert observation.position == (2, 2)
    assert observation.battery == 15


def test_observation_sees_nearby_dirty_cells():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset({
            (2, 3),
        }),
        battery=15,
    )

    observation = environment.get_observation(state)

    assert (2, 3) in observation.dirty_cells


def test_observation_hides_distant_dirty_cells():
    environment = CleaningEnvironment(
        rows=10,
        cols=10,
        observation_radius=1,
    )

    state = State(
        position=(5, 5),
        dirty_cells=frozenset({
            (5, 7),
        }),
        battery=15,
    )

    observation = environment.get_observation(state)

    assert (5, 7) not in observation.dirty_cells


def test_observation_sees_nearby_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        dynamic_obstacles=frozenset({
            (2, 3),
        }),
        observation_radius=1,
    )

    state = State(
        position=(2, 2),
        dirty_cells=frozenset(),
        battery=15,
    )

    observation = environment.get_observation(state)

    assert (2, 3) in observation.dynamic_obstacles


def test_observation_hides_distant_dynamic_obstacle():
    environment = CleaningEnvironment(
        rows=10,
        cols=10,
        dynamic_obstacles=frozenset({
            (5, 7),
        }),
        observation_radius=1,
    )

    state = State(
        position=(5, 5),
        dirty_cells=frozenset(),
        battery=15,
    )

    observation = environment.get_observation(state)

    assert (5, 7) not in observation.dynamic_obstacles


def test_observation_radius_two_sees_farther_cells():
    environment = CleaningEnvironment(
        rows=10,
        cols=10,
        observation_radius=2,
    )

    state = State(
        position=(5, 5),
        dirty_cells=frozenset({
            (5, 7),
        }),
        battery=15,
    )

    observation = environment.get_observation(state)

    assert (5, 7) in observation.dirty_cells


def test_observation_does_not_modify_state():
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
    )

    dirty_cells = frozenset({
        (2, 3),
    })

    state = State(
        position=(2, 2),
        dirty_cells=dirty_cells,
        battery=15,
    )

    observation = environment.get_observation(state)

    assert state.dirty_cells == dirty_cells
    assert state.battery == 15
    assert state.position == (2, 2)
    assert observation is not state