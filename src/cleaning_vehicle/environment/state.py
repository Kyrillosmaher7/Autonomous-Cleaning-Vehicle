
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class State:
    """
    Immutable representation of the cleaning vehicle's RL state.

    The state contains only information that describes the
    current situation of the agent:

        - robot position
        - remaining dirty cells

    Static environment information such as obstacles belongs
    to the Grid and is not stored in the state for Version 1.

    The state is immutable and hashable, making it suitable
    for use as a key in a Q-table.
    """

    position: Position
    dirty_cells: FrozenSet[Position]

    def __post_init__(self) -> None:
        """Validate and normalize the state."""

        row, column = self.position

        if not isinstance(row, int) or not isinstance(column, int):
            raise TypeError(
                "Position must contain two integers."
            )

        if row < 0 or column < 0:
            raise ValueError(
                "Position coordinates cannot be negative."
            )

        # Ensure dirty_cells is always a frozenset.
        object.__setattr__(
            self,
            "dirty_cells",
            frozenset(self.dirty_cells),
        )

        # Validate every dirty-cell position.
        for dirty_position in self.dirty_cells:
            if (
                not isinstance(dirty_position, tuple)
                or len(dirty_position) != 2
            ):
                raise TypeError(
                    "Every dirty cell must be a (row, column) tuple."
                )

            dirty_row, dirty_column = dirty_position

            if (
                not isinstance(dirty_row, int)
                or not isinstance(dirty_column, int)
            ):
                raise TypeError(
                    "Dirty cell coordinates must be integers."
                )

            if dirty_row < 0 or dirty_column < 0:
                raise ValueError(
                    "Dirty cell coordinates cannot be negative."
                )


    @property
    def row(self) -> int:
        """Return the robot's current row."""

        return self.position[0]

    @property
    def column(self) -> int:
        """Return the robot's current column."""

        return self.position[1]


    @property
    def remaining_dirty(self) -> int:
        """Return the number of dirty cells remaining."""

        return len(self.dirty_cells)

    @property
    def is_clean(self) -> bool:
        """Return True when all dirty cells have been cleaned."""

        return len(self.dirty_cells) == 0

    def is_dirty(self, position: Position) -> bool:
        """Return True if the specified position is dirty."""

        return position in self.dirty_cells


    def move(self, position: Position) -> State:
        """
        Return a new state with the robot at a new position.

        The current state is not modified.
        """

        return State(
            position=position,
            dirty_cells=self.dirty_cells,
        )

    def clean(self, position: Position) -> State:
        """
        Return a new state where the specified dirty cell
        has been cleaned.

        The current state is not modified.

        If the position is not dirty, the returned state
        is equivalent to the current state.
        """

        if position not in self.dirty_cells:
            return self

        remaining_dirty = self.dirty_cells - {position}

        return State(
            position=self.position,
            dirty_cells=remaining_dirty,
        )

    def move_and_clean(self, position: Position) -> State:
        """
        Return a new state where:

            1. The robot moves to `position`.
            2. The cell is cleaned if it is dirty.

        The current state is not modified.
        """

        remaining_dirty = self.dirty_cells - {position}

        return State(
            position=position,
            dirty_cells=remaining_dirty,
        )

    # ==========================================================
    # Q-Learning Representation
    # ==========================================================

    def as_tuple(self) -> tuple[Position, FrozenSet[Position]]:
        """
        Return the state as a tuple.

        This gives us an explicit representation that can
        be used by future Q-Learning components.
        """

        return (
            self.position,
            self.dirty_cells,
        )

    def __hash__(self) -> int:
        """
        Return a hash based on the complete state.

        This allows:

            Q[state]

        to work with a dictionary-based Q-table.
        """

        return hash(
            (
                self.position,
                self.dirty_cells,
            )
        )

    def __repr__(self) -> str:
        return (
            f"State("
            f"position={self.position}, "
            f"dirty_cells={self.dirty_cells}"
            f")"
        )
