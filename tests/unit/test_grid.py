
import pytest

from  cleaning_vehicle.environment.grid import Grid


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def empty_grid():
    """Create a simple empty 5x5 grid."""
    return Grid(width=5, height=5)


@pytest.fixture
def sample_grid():
    """
    Create a grid containing both obstacles and dirty cells.

    Layout:

        . . . . .
        . D D . .
        . . # . .
        . . # D .
        . . . . .
    """
    return Grid(
        width=5,
        height=5,
        obstacles={
            (2, 2),
            (3, 2),
        },
        dirty_cells={
            (1, 1),
            (1, 2),
            (3, 3),
        },
    )


# ============================================================
# Initialization Tests
# ============================================================

class TestGridInitialization:

    def test_create_empty_grid(self):
        grid = Grid(width=5, height=4)

        assert grid.width == 5
        assert grid.height == 4
        assert grid.obstacles == set()
        assert grid.dirty_cells == set()

    def test_create_grid_with_obstacles(self):
        obstacles = {
            (1, 1),
            (2, 3),
        }

        grid = Grid(
            width=5,
            height=5,
            obstacles=obstacles,
        )

        assert grid.obstacles == obstacles

    def test_create_grid_with_dirty_cells(self):
        dirty_cells = {
            (0, 0),
            (1, 2),
            (4, 4),
        }

        grid = Grid(
            width=5,
            height=5,
            dirty_cells=dirty_cells,
        )

        assert grid.dirty_cells == dirty_cells

    def test_create_grid_with_obstacles_and_dirty_cells(self):
        obstacles = {(1, 1)}
        dirty_cells = {(2, 2)}

        grid = Grid(
            width=5,
            height=5,
            obstacles=obstacles,
            dirty_cells=dirty_cells,
        )

        assert grid.obstacles == obstacles
        assert grid.dirty_cells == dirty_cells

    def test_width_must_be_positive(self):
        with pytest.raises(ValueError):
            Grid(width=0, height=5)

    def test_negative_width_is_invalid(self):
        with pytest.raises(ValueError):
            Grid(width=-1, height=5)

    def test_height_must_be_positive(self):
        with pytest.raises(ValueError):
            Grid(width=5, height=0)

    def test_negative_height_is_invalid(self):
        with pytest.raises(ValueError):
            Grid(width=5, height=-1)


# ============================================================
# Position Validation Tests
# ============================================================

class TestPositionValidation:

    def test_valid_position_inside_grid(self, empty_grid):
        assert empty_grid.is_valid_position((0, 0)) is True

    def test_valid_position_at_last_cell(self, empty_grid):
        assert empty_grid.is_valid_position((4, 4)) is True

    def test_valid_position_in_middle(self, empty_grid):
        assert empty_grid.is_valid_position((2, 3)) is True

    def test_negative_row_is_invalid(self, empty_grid):
        assert empty_grid.is_valid_position((-1, 2)) is False

    def test_negative_column_is_invalid(self, empty_grid):
        assert empty_grid.is_valid_position((2, -1)) is False

    def test_row_equal_to_height_is_invalid(self, empty_grid):
        assert empty_grid.is_valid_position((5, 2)) is False

    def test_column_equal_to_width_is_invalid(self, empty_grid):
        assert empty_grid.is_valid_position((2, 5)) is False

    def test_position_beyond_grid_is_invalid(self, empty_grid):
        assert empty_grid.is_valid_position((10, 10)) is False


# ============================================================
# Obstacle Tests
# ============================================================

class TestObstacles:

    def test_obstacle_is_detected(self, sample_grid):
        assert sample_grid.is_obstacle((2, 2)) is True

    def test_second_obstacle_is_detected(self, sample_grid):
        assert sample_grid.is_obstacle((3, 2)) is True

    def test_non_obstacle_is_detected(self, sample_grid):
        assert sample_grid.is_obstacle((0, 0)) is False

    def test_dirty_cell_is_not_obstacle(self, sample_grid):
        assert sample_grid.is_obstacle((1, 1)) is False

    def test_obstacle_outside_grid_is_rejected(self):
        with pytest.raises(ValueError):
            Grid(
                width=5,
                height=5,
                obstacles={(5, 5)},
            )

    def test_negative_obstacle_position_is_rejected(self):
        with pytest.raises(ValueError):
            Grid(
                width=5,
                height=5,
                obstacles={(-1, 0)},
            )


# ============================================================
# Dirty Cell Tests
# ============================================================

class TestDirtyCells:

    def test_dirty_cell_is_detected(self, sample_grid):
        assert sample_grid.is_dirty((1, 1)) is True

    def test_second_dirty_cell_is_detected(self, sample_grid):
        assert sample_grid.is_dirty((1, 2)) is True

    def test_third_dirty_cell_is_detected(self, sample_grid):
        assert sample_grid.is_dirty((3, 3)) is True

    def test_clean_cell_is_not_dirty(self, sample_grid):
        assert sample_grid.is_dirty((0, 0)) is False

    def test_dirty_cell_outside_grid_is_rejected(self):
        with pytest.raises(ValueError):
            Grid(
                width=5,
                height=5,
                dirty_cells={(5, 5)},
            )

    def test_negative_dirty_position_is_rejected(self):
        with pytest.raises(ValueError):
            Grid(
                width=5,
                height=5,
                dirty_cells={(-1, 0)},
            )


# ============================================================
# Obstacle / Dirty Collision Tests
# ============================================================

class TestInvalidCellConfiguration:

    def test_cell_cannot_be_both_obstacle_and_dirty(self):
        with pytest.raises(ValueError):
            Grid(
                width=5,
                height=5,
                obstacles={(2, 2)},
                dirty_cells={(2, 2)},
            )

    def test_multiple_overlapping_cells_are_rejected(self):
        with pytest.raises(ValueError):
            Grid(
                width=5,
                height=5,
                obstacles={
                    (1, 1),
                    (2, 2),
                },
                dirty_cells={
                    (2, 2),
                    (3, 3),
                },
            )


# ============================================================
# Traversability Tests
# ============================================================

class TestTraversability:

    def test_empty_cell_is_traversable(self, sample_grid):
        assert sample_grid.is_traversable((0, 0)) is True

    def test_dirty_cell_is_traversable(self, sample_grid):
        assert sample_grid.is_traversable((1, 1)) is True

    def test_obstacle_is_not_traversable(self, sample_grid):
        assert sample_grid.is_traversable((2, 2)) is False

    def test_second_obstacle_is_not_traversable(self, sample_grid):
        assert sample_grid.is_traversable((3, 2)) is False

    def test_position_outside_grid_is_not_traversable(self, sample_grid):
        assert sample_grid.is_traversable((5, 5)) is False

    def test_negative_position_is_not_traversable(self, sample_grid):
        assert sample_grid.is_traversable((-1, 0)) is False


# ============================================================
# Cell Type Tests
# ============================================================

class TestGetCell:

    def test_empty_cell_returns_empty(self, sample_grid):
        assert sample_grid.get_cell((0, 0)) == Grid.EMPTY

    def test_dirty_cell_returns_dirty(self, sample_grid):
        assert sample_grid.get_cell((1, 1)) == Grid.DIRTY

    def test_obstacle_returns_obstacle(self, sample_grid):
        assert sample_grid.get_cell((2, 2)) == Grid.OBSTACLE

    def test_invalid_position_raises_error(self, sample_grid):
        with pytest.raises(ValueError):
            sample_grid.get_cell((5, 5))


# ============================================================
# Cleaning Tests
# ============================================================

class TestCleaning:

    def test_clean_dirty_cell(self, sample_grid):
        result = sample_grid.clean_cell((1, 1))

        assert result is True
        assert sample_grid.is_dirty((1, 1)) is False

    def test_cleaning_removes_cell_from_dirty_set(self, sample_grid):
        sample_grid.clean_cell((1, 1))

        assert (1, 1) not in sample_grid.dirty_cells

    def test_cleaning_decreases_dirty_count(self, sample_grid):
        initial_count = sample_grid.total_dirty_cells

        sample_grid.clean_cell((1, 1))

        assert sample_grid.total_dirty_cells == initial_count - 1

    def test_cleaning_clean_cell_returns_false(self, sample_grid):
        result = sample_grid.clean_cell((0, 0))

        assert result is False

    def test_cleaning_same_cell_twice(self, sample_grid):
        first_result = sample_grid.clean_cell((1, 1))
        second_result = sample_grid.clean_cell((1, 1))

        assert first_result is True
        assert second_result is False

    def test_cleaning_obstacle_returns_false(self, sample_grid):
        result = sample_grid.clean_cell((2, 2))

        assert result is False

    def test_cleaning_invalid_position_raises_error(self, sample_grid):
        with pytest.raises(ValueError):
            sample_grid.clean_cell((5, 5))


# ============================================================
# Statistics Tests
# ============================================================

class TestStatistics:

    def test_total_cells(self, empty_grid):
        assert empty_grid.total_cells == 25

    def test_total_cells_for_rectangular_grid(self):
        grid = Grid(width=7, height=3)

        assert grid.total_cells == 21

    def test_total_dirty_cells(self, sample_grid):
        assert sample_grid.total_dirty_cells == 3

    def test_total_obstacles(self, sample_grid):
        assert sample_grid.total_obstacles == 2

    def test_empty_grid_is_all_clean(self, empty_grid):
        assert empty_grid.all_clean is True

    def test_grid_with_dirty_cells_is_not_all_clean(self, sample_grid):
        assert sample_grid.all_clean is False

    def test_grid_becomes_all_clean_after_cleaning(self):
        grid = Grid(
            width=3,
            height=3,
            dirty_cells={
                (0, 0),
                (1, 1),
            },
        )

        grid.clean_cell((0, 0))
        assert grid.all_clean is False

        grid.clean_cell((1, 1))
        assert grid.all_clean is True


# ============================================================
# Reset Tests
# ============================================================

class TestReset:

    def test_reset_replaces_obstacles(self):
        grid = Grid(
            width=5,
            height=5,
            obstacles={(1, 1)},
            dirty_cells={(2, 2)},
        )

        grid.reset(
            obstacles={(3, 3)},
            dirty_cells={(4, 4)},
        )

        assert grid.obstacles == {(3, 3)}
        assert grid.dirty_cells == {(4, 4)}

    def test_reset_removes_old_obstacles(self):
        grid = Grid(
            width=5,
            height=5,
            obstacles={(1, 1)},
        )

        grid.reset(obstacles={(3, 3)})

        assert (1, 1) not in grid.obstacles
        assert (3, 3) in grid.obstacles

    def test_reset_removes_old_dirty_cells(self):
        grid = Grid(
            width=5,
            height=5,
            dirty_cells={(1, 1)},
        )

        grid.reset(dirty_cells={(3, 3)})

        assert (1, 1) not in grid.dirty_cells
        assert (3, 3) in grid.dirty_cells

    def test_reset_without_arguments_keeps_configuration(self):
        obstacles = {(1, 1)}
        dirty_cells = {(2, 2)}

        grid = Grid(
            width=5,
            height=5,
            obstacles=obstacles,
            dirty_cells=dirty_cells,
        )

        grid.reset()

        assert grid.obstacles == obstacles
        assert grid.dirty_cells == dirty_cells

    def test_reset_validates_new_positions(self):
        grid = Grid(width=5, height=5)

        with pytest.raises(ValueError):
            grid.reset(obstacles={(5, 5)})

    def test_reset_validates_overlap(self):
        grid = Grid(width=5, height=5)

        with pytest.raises(ValueError):
            grid.reset(
                obstacles={(2, 2)},
                dirty_cells={(2, 2)},
            )


# ============================================================
# Render Tests
# ============================================================

class TestRender:

    def test_render_empty_grid(self, empty_grid, capsys):
        empty_grid.render()

        captured = capsys.readouterr()

        expected = (
            ". . . . .\n"
            ". . . . .\n"
            ". . . . .\n"
            ". . . . .\n"
            ". . . . .\n"
            "\n"
        )

        assert captured.out == expected

    def test_render_obstacles(self, capsys):
        grid = Grid(
            width=3,
            height=3,
            obstacles={(1, 1)},
        )

        grid.render()

        captured = capsys.readouterr()

        expected = (
            ". . .\n"
            ". # .\n"
            ". . .\n"
            "\n"
        )

        assert captured.out == expected

    def test_render_dirty_cells(self, capsys):
        grid = Grid(
            width=3,
            height=3,
            dirty_cells={(0, 0), (2, 2)},
        )

        grid.render()

        captured = capsys.readouterr()

        expected = (
            "D . .\n"
            ". . .\n"
            ". . D\n"
            "\n"
        )

        assert captured.out == expected

    def test_render_mixed_environment(self, capsys):
        grid = Grid(
            width=3,
            height=3,
            obstacles={(1, 1)},
            dirty_cells={(0, 0), (2, 2)},
        )

        grid.render()

        captured = capsys.readouterr()

        expected = (
            "D . .\n"
            ". # .\n"
            ". . D\n"
            "\n"
        )

        assert captured.out == expected


# ============================================================
# Representation Tests
# ============================================================

class TestRepresentation:

    def test_repr(self):
        grid = Grid(
            width=5,
            height=4,
            obstacles={(1, 1), (2, 2)},
            dirty_cells={(0, 0), (3, 3), (1, 3)},
        )

        result = repr(grid)

        expected = (
            "Grid("
            "width=5, "
            "height=4, "
            "obstacles=2, "
            "dirty_cells=3"
            ")"
        )

        assert result == expected


# ============================================================
# Boundary Tests
# ============================================================

class TestBoundaries:

    def test_top_left_corner(self, empty_grid):
        assert empty_grid.is_valid_position((0, 0)) is True

    def test_top_right_corner(self, empty_grid):
        assert empty_grid.is_valid_position((0, 4)) is True

    def test_bottom_left_corner(self, empty_grid):
        assert empty_grid.is_valid_position((4, 0)) is True

    def test_bottom_right_corner(self, empty_grid):
        assert empty_grid.is_valid_position((4, 4)) is True

    def test_position_above_grid(self, empty_grid):
        assert empty_grid.is_valid_position((-1, 2)) is False

    def test_position_below_grid(self, empty_grid):
        assert empty_grid.is_valid_position((5, 2)) is False

    def test_position_left_of_grid(self, empty_grid):
        assert empty_grid.is_valid_position((2, -1)) is False

    def test_position_right_of_grid(self, empty_grid):
        assert empty_grid.is_valid_position((2, 5)) is False


# ============================================================
# Integration-Level Grid Behavior
# ============================================================

class TestGridBehavior:

    def test_complete_cleaning_workflow(self):
        grid = Grid(
            width=4,
            height=4,
            obstacles={
                (1, 1),
                (2, 2),
            },
            dirty_cells={
                (0, 0),
                (0, 3),
                (3, 3),
            },
        )

        # Initial state
        assert grid.total_cells == 16
        assert grid.total_obstacles == 2
        assert grid.total_dirty_cells == 3
        assert grid.all_clean is False

        # Robot can theoretically traverse dirty cells
        assert grid.is_traversable((0, 0)) is True
        assert grid.is_traversable((0, 3)) is True

        # Robot cannot traverse obstacles
        assert grid.is_traversable((1, 1)) is False
        assert grid.is_traversable((2, 2)) is False

        # Clean all dirty cells
        assert grid.clean_cell((0, 0)) is True
        assert grid.clean_cell((0, 3)) is True
        assert grid.clean_cell((3, 3)) is True

        # Final state
        assert grid.total_dirty_cells == 0
        assert grid.all_clean is True

    def test_grid_does_not_modify_input_sets(self):
        obstacles = {(1, 1)}
        dirty_cells = {(2, 2)}

        Grid(
            width=5,
            height=5,
            obstacles=obstacles,
            dirty_cells=dirty_cells,
        )

        assert obstacles == {(1, 1)}
        assert dirty_cells == {(2, 2)}

    def test_dirty_cell_remains_traversable_after_cleaning(self):
        grid = Grid(
            width=3,
            height=3,
            dirty_cells={(1, 1)},
        )

        assert grid.is_traversable((1, 1)) is True

        grid.clean_cell((1, 1))

        assert grid.is_traversable((1, 1)) is True
        assert grid.is_dirty((1, 1)) is False

    def test_obstacle_remains_obstacle_after_clean_attempt(self):
        grid = Grid(
            width=3,
            height=3,
            obstacles={(1, 1)},
        )

        result = grid.clean_cell((1, 1))

        assert result is False
        assert grid.is_obstacle((1, 1)) is True
        assert grid.is_traversable((1, 1)) is False

