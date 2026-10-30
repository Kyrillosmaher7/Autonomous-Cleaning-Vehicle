from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class State:
    """
    Immutable representation of the cleaning vehicle's RL state.

    Dynamic information:

        - robot position
        - remaining dirty cells
        - remaining battery

    Static environment information such as obstacles and the
    charging station belongs to the environment.
    """

    position: Position
    dirty_cells: FrozenSet[Position]
    battery: int = 20

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

        if not isinstance(self.battery, int):
            raise TypeError(
                "Battery must be an integer."
            )

        if self.battery < 0:
            raise ValueError(
                "Battery cannot be negative."
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


    @property
    def battery_empty(self) -> bool:
        """Return True when the battery is empty."""

        return self.battery == 0

    def consume_battery(self, amount: int = 1) -> State:
        """
        Return a new state with reduced battery.

        Battery can never become negative.
        """

        if amount < 0:
            raise ValueError(
                "Battery consumption amount cannot be negative."
            )

        new_battery = max(0, self.battery - amount)

        return State(
            position=self.position,
            dirty_cells=self.dirty_cells,
            battery=new_battery,
        )

    def recharge(self, max_battery: int) -> State:
        """
        Return a new state with a fully charged battery.
        """

        if max_battery <= 0:
            raise ValueError(
                "max_battery must be positive."
            )

        return State(
            position=self.position,
            dirty_cells=self.dirty_cells,
            battery=max_battery,
        )



    def move(self, position: Position) -> State:
        """
        Return a new state with the robot at a new position.

        The current state is not modified.
        """

        return State(
            position=position,
            dirty_cells=self.dirty_cells,
            battery=self.battery,
        )

    def clean(self, position: Position) -> State:
        """
        Return a new state where the specified dirty cell
        has been cleaned.
        """

        if position not in self.dirty_cells:
            return self

        remaining_dirty = self.dirty_cells - {position}

        return State(
            position=self.position,
            dirty_cells=remaining_dirty,
            battery=self.battery,
        )

    def move_and_clean(self, position: Position) -> State:
        """
        Return a new state where:

            1. The robot moves to position.
            2. The cell is cleaned if dirty.

        Battery is unchanged here because battery consumption
        belongs to the environment transition.
        """

        remaining_dirty = self.dirty_cells - {position}

        return State(
            position=position,
            dirty_cells=remaining_dirty,
            battery=self.battery,
        )


    def as_tuple(
        self,
    ) -> tuple[Position, FrozenSet[Position], int]:
        """
        Return the complete dynamic state representation.

        Battery is included because it affects future decisions.
        """

        return (
            self.position,
            self.dirty_cells,
            self.battery,
        )

    def __hash__(self) -> int:
        """Return a hash based on the complete state."""

        return hash(
            (
                self.position,
                self.dirty_cells,
                self.battery,
            )
        )

    def __repr__(self) -> str:
        return (
            f"State("
            f"position={self.position}, "
            f"dirty_cells={self.dirty_cells}, "
            f"battery={self.battery}"
            f")"
        )