
import pytest

from cleaning_vehicle.environment.state import State


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def clean_state():
    """State with no remaining dirty cells."""
    return State(
        position=(2, 2),
        dirty_cells=frozenset(),
    )


@pytest.fixture
def dirty_state():
    """State with multiple dirty cells."""
    return State(
        position=(2, 2),
        dirty_cells=frozenset({
            (0, 0),
            (1, 3),
            (4, 4),
        }),
    )


# ============================================================
# Initialization
# ============================================================

class TestStateInitialization:

    def test_create_state(self):
        state = State(
            position=(2, 3),
            dirty_cells=frozenset({
                (1, 1),
                (4, 4),
            }),
        )

        assert state.position == (2, 3)
        assert state.dirty_cells == frozenset({
            (1, 1),
            (4, 4),
        })

    def test_create_state_with_set(self):
        """
        State should normalize a mutable set into frozenset.
        """
        state = State(
            position=(1, 2),
            dirty_cells={
                (0, 0),
                (3, 3),
            },
        )

        assert isinstance(state.dirty_cells, frozenset)
        assert state.dirty_cells == frozenset({
            (0, 0),
            (3, 3),
        })

    def test_create_state_with_empty_dirty_cells(self):
        state = State(
            position=(0, 0),
            dirty_cells=frozenset(),
        )

        assert state.dirty_cells == frozenset()

    def test_position_must_contain_integers(self):
        with pytest.raises(TypeError):
            State(
                position=("1", 2),
                dirty_cells=frozenset(),
            )

    def test_position_negative_row_is_invalid(self):
        with pytest.raises(ValueError):
            State(
                position=(-1, 2),
                dirty_cells=frozenset(),
            )

    def test_position_negative_column_is_invalid(self):
        with pytest.raises(ValueError):
            State(
                position=(2, -1),
                dirty_cells=frozenset(),
            )


# ============================================================
# Dirty Cell Validation
# ============================================================

class TestDirtyCellValidation:

    def test_dirty_cell_must_be_tuple(self):
        with pytest.raises(TypeError):
            State(
                position=(0, 0),
                dirty_cells=frozenset({
                    [1, 2],
                }),
            )

    def test_dirty_cell_must_have_two_coordinates(self):
        with pytest.raises(TypeError):
            State(
                position=(0, 0),
                dirty_cells=frozenset({
                    (1, 2, 3),
                }),
            )

    def test_dirty_cell_coordinates_must_be_integers(self):
        with pytest.raises(TypeError):
            State(
                position=(0, 0),
                dirty_cells=frozenset({
                    ("1", 2),
                }),
            )

    def test_negative_dirty_row_is_invalid(self):
        with pytest.raises(ValueError):
            State(
                position=(0, 0),
                dirty_cells=frozenset({
                    (-1, 2),
                }),
            )

    def test_negative_dirty_column_is_invalid(self):
        with pytest.raises(ValueError):
            State(
                position=(0, 0),
                dirty_cells=frozenset({
                    (2, -1),
                }),
            )

    def test_multiple_valid_dirty_cells(self):
        dirty_cells = frozenset({
            (0, 0),
            (1, 1),
            (2, 2),
        })

        state = State(
            position=(3, 3),
            dirty_cells=dirty_cells,
        )

        assert state.dirty_cells == dirty_cells


# ============================================================
# Position Properties
# ============================================================

class TestPositionProperties:

    def test_row(self, dirty_state):
        assert dirty_state.row == 2

    def test_column(self, dirty_state):
        assert dirty_state.column == 2

    def test_position(self, dirty_state):
        assert dirty_state.position == (2, 2)


# ============================================================
# Dirty Cell Properties
# ============================================================

class TestDirtyCellProperties:

    def test_remaining_dirty(self, dirty_state):
        assert dirty_state.remaining_dirty == 3

    def test_remaining_dirty_empty(self, clean_state):
        assert clean_state.remaining_dirty == 0

    def test_is_clean_when_no_dirty_cells(self, clean_state):
        assert clean_state.is_clean is True

    def test_is_clean_when_dirty_cells_exist(self, dirty_state):
        assert dirty_state.is_clean is False

    def test_is_dirty_for_dirty_position(self, dirty_state):
        assert dirty_state.is_dirty((0, 0)) is True

    def test_is_dirty_for_another_dirty_position(self, dirty_state):
        assert dirty_state.is_dirty((4, 4)) is True

    def test_is_dirty_for_clean_position(self, dirty_state):
        assert dirty_state.is_dirty((2, 2)) is False


# ============================================================
# Immutability
# ============================================================

class TestImmutability:

    def test_position_cannot_be_modified(self, dirty_state):
        with pytest.raises((AttributeError, TypeError)):
            dirty_state.position = (3, 3)

    def test_dirty_cells_cannot_be_modified(self, dirty_state):
        with pytest.raises((AttributeError, TypeError)):
            dirty_state.dirty_cells = frozenset()

    def test_frozenset_cannot_be_modified(self, dirty_state):
        with pytest.raises(AttributeError):
            dirty_state.dirty_cells.add((5, 5))


# ============================================================
# Equality
# ============================================================

class TestEquality:

    def test_equal_states_are_equal(self):
        state_a = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (0, 0),
                (3, 3),
            }),
        )

        state_b = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (0, 0),
                (3, 3),
            }),
        )

        assert state_a == state_b

    def test_different_positions_are_not_equal(self):
        state_a = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (0, 0),
            }),
        )

        state_b = State(
            position=(2, 3),
            dirty_cells=frozenset({
                (0, 0),
            }),
        )

        assert state_a != state_b

    def test_different_dirty_cells_are_not_equal(self):
        state_a = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (0, 0),
            }),
        )

        state_b = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (1, 1),
            }),
        )

        assert state_a != state_b


# ============================================================
# Hashing
# ============================================================

class TestHashing:

    def test_state_is_hashable(self, dirty_state):
        result = hash(dirty_state)

        assert isinstance(result, int)

    def test_equal_states_have_same_hash(self):
        state_a = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (0, 0),
                (3, 3),
            }),
        )

        state_b = State(
            position=(2, 2),
            dirty_cells=frozenset({
                (0, 0),
                (3, 3),
            }),
        )

        assert hash(state_a) == hash(state_b)

    def test_state_can_be_used_as_dictionary_key(self, dirty_state):
        q_table = {}

        q_table[dirty_state] = 10.5

        assert q_table[dirty_state] == 10.5

    def test_state_can_be_used_in_set(self, dirty_state):
        states = {dirty_state}

        assert dirty_state in states


# ============================================================
# move()
# ============================================================

class TestMove:

    def test_move_changes_position(self, dirty_state):
        new_state = dirty_state.move((3, 3))

        assert new_state.position == (3, 3)

    def test_move_preserves_dirty_cells(self, dirty_state):
        new_state = dirty_state.move((3, 3))

        assert new_state.dirty_cells == dirty_state.dirty_cells

    def test_move_does_not_modify_original_state(self, dirty_state):
        original_position = dirty_state.position

        dirty_state.move((4, 4))

        assert dirty_state.position == original_position

    def test_move_returns_new_state(self, dirty_state):
        new_state = dirty_state.move((3, 3))

        assert isinstance(new_state, State)
        assert new_state is not dirty_state

    def test_move_to_same_position(self, dirty_state):
        new_state = dirty_state.move(dirty_state.position)

        assert new_state == dirty_state


# ============================================================
# clean()
# ============================================================

class TestClean:

    def test_clean_dirty_cell(self, dirty_state):
        new_state = dirty_state.clean((0, 0))

        assert (0, 0) not in new_state.dirty_cells

    def test_clean_reduces_dirty_count(self, dirty_state):
        new_state = dirty_state.clean((0, 0))

        assert new_state.remaining_dirty == 2

    def test_clean_preserves_position(self, dirty_state):
        new_state = dirty_state.clean((0, 0))

        assert new_state.position == dirty_state.position

    def test_clean_does_not_modify_original_state(self, dirty_state):
        dirty_state.clean((0, 0))

        assert (0, 0) in dirty_state.dirty_cells

    def test_clean_clean_cell_returns_same_state(self, dirty_state):
        new_state = dirty_state.clean((2, 2))

        assert new_state is dirty_state

    def test_clean_last_dirty_cell(self):
        state = State(
            position=(1, 1),
            dirty_cells=frozenset({
                (1, 1),
            }),
        )

        new_state = state.clean((1, 1))

        assert new_state.is_clean is True
        assert new_state.remaining_dirty == 0


# ============================================================
# move_and_clean()
# ============================================================

class TestMoveAndClean:

    def test_move_and_clean_changes_position(self, dirty_state):
        new_state = dirty_state.move_and_clean((0, 0))

        assert new_state.position == (0, 0)

    def test_move_and_clean_removes_dirty_cell(self, dirty_state):
        new_state = dirty_state.move_and_clean((0, 0))

        assert (0, 0) not in new_state.dirty_cells

    def test_move_and_clean_reduces_dirty_count(self, dirty_state):
        new_state = dirty_state.move_and_clean((0, 0))

        assert new_state.remaining_dirty == 2

    def test_move_and_clean_preserves_other_dirty_cells(self, dirty_state):
        new_state = dirty_state.move_and_clean((0, 0))

        assert new_state.dirty_cells == frozenset({
            (1, 3),
            (4, 4),
        })

    def test_move_and_clean_does_not_modify_original_state(self, dirty_state):
        dirty_state.move_and_clean((0, 0))

        assert dirty_state.position == (2, 2)
        assert (0, 0) in dirty_state.dirty_cells

    def test_move_and_clean_on_clean_cell(self, dirty_state):
        new_state = dirty_state.move_and_clean((2, 2))

        assert new_state.position == (2, 2)
        assert new_state.dirty_cells == dirty_state.dirty_cells

    def test_move_and_clean_last_dirty_cell(self):
        state = State(
            position=(0, 0),
            dirty_cells=frozenset({
                (2, 2),
            }),
        )

        new_state = state.move_and_clean((2, 2))

        assert new_state.position == (2, 2)
        assert new_state.is_clean is True


# ============================================================
# as_tuple()
# ============================================================

class TestAsTuple:

    def test_as_tuple_returns_tuple(self, dirty_state):
        result = dirty_state.as_tuple()

        assert isinstance(result, tuple)

    def test_as_tuple_contains_position(self, dirty_state):
        result = dirty_state.as_tuple()

        assert result[0] == dirty_state.position

    def test_as_tuple_contains_dirty_cells(self, dirty_state):
        result = dirty_state.as_tuple()

        assert result[1] == dirty_state.dirty_cells

    def test_as_tuple_matches_state(self, dirty_state):
        assert dirty_state.as_tuple() == (
            dirty_state.position,
            dirty_state.dirty_cells,
        )


# ============================================================
# Representation
# ============================================================

class TestRepresentation:

    def test_repr(self):
        state = State(
            position=(2, 3),
            dirty_cells=frozenset({
                (0, 0),
                (4, 4),
            }),
        )

        result = repr(state)

        assert result.startswith("State(")
        assert "position=(2, 3)" in result
        assert "dirty_cells=" in result


# ============================================================
# State Transition Integration
# ============================================================

class TestStateTransitions:

    def test_move_then_clean(self):
        state = State(
            position=(0, 0),
            dirty_cells=frozenset({
                (1, 1),
                (2, 2),
            }),
        )

        moved_state = state.move((1, 1))
        cleaned_state = moved_state.clean((1, 1))

        assert state.position == (0, 0)
        assert state.remaining_dirty == 2

        assert moved_state.position == (1, 1)
        assert moved_state.remaining_dirty == 2

        assert cleaned_state.position == (1, 1)
        assert cleaned_state.remaining_dirty == 1
        assert (1, 1) not in cleaned_state.dirty_cells

    def test_multiple_cleaning_transitions(self):
        state = State(
            position=(0, 0),
            dirty_cells=frozenset({
                (1, 1),
                (2, 2),
                (3, 3),
            }),
        )

        state = state.move_and_clean((1, 1))
        assert state.remaining_dirty == 2

        state = state.move_and_clean((2, 2))
        assert state.remaining_dirty == 1

        state = state.move_and_clean((3, 3))
        assert state.remaining_dirty == 0
        assert state.is_clean is True

    def test_original_state_remains_unchanged_through_chain(self):
        original = State(
            position=(0, 0),
            dirty_cells=frozenset({
                (1, 1),
                (2, 2),
            }),
        )

        state = original.move_and_clean((1, 1))
        state = state.move_and_clean((2, 2))

        assert original.position == (0, 0)
        assert original.dirty_cells == frozenset({
            (1, 1),
            (2, 2),
        })
        assert original.remaining_dirty == 2

        assert state.position == (2, 2)
        assert state.is_clean is True

