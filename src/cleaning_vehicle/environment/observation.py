from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from .state import Position


@dataclass(frozen=True, slots=True)
class Observation:
    """
    Immutable partial observation available to the agent.

    Unlike State, an Observation does not necessarily describe
    the complete environment.

    It contains only the information that the agent is allowed
    to observe:

        - robot position
        - battery
        - visible dirty cells
        - visible dynamic obstacles
    """

    position: Position
    battery: int
    dirty_cells: FrozenSet[Position]
    dynamic_obstacles: FrozenSet[Position]

    def __post_init__(self) -> None:
        """Validate and normalize the observation."""

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

        object.__setattr__(
            self,
            "dirty_cells",
            frozenset(self.dirty_cells),
        )

        object.__setattr__(
            self,
            "dynamic_obstacles",
            frozenset(self.dynamic_obstacles),
        )

        self._validate_positions(
            self.dirty_cells,
            "dirty cell",
        )

        self._validate_positions(
            self.dynamic_obstacles,
            "dynamic obstacle",
        )

    @staticmethod
    def _validate_positions(
        positions: FrozenSet[Position],
        name: str,
    ) -> None:
        """Validate a collection of positions."""

        for position in positions:

            if (
                not isinstance(position, tuple)
                or len(position) != 2
            ):
                raise TypeError(
                    f"Every {name} must be a "
                    "(row, column) tuple."
                )

            row, column = position

            if (
                not isinstance(row, int)
                or not isinstance(column, int)
            ):
                raise TypeError(
                    f"{name.capitalize()} coordinates "
                    "must be integers."
                )

            if row < 0 or column < 0:
                raise ValueError(
                    f"{name.capitalize()} coordinates "
                    "cannot be negative."
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
        """Return the number of visible dirty cells."""

        return len(self.dirty_cells)

    def is_dirty(self, position: Position) -> bool:
        """Return True when a visible cell is dirty."""

        return position in self.dirty_cells

    def is_dynamic_obstacle(
        self,
        position: Position,
    ) -> bool:
        """Return True when a dynamic obstacle is visible."""

        return position in self.dynamic_obstacles

    @property
    def battery_empty(self) -> bool:
        """Return True when the battery is empty."""

        return self.battery == 0

    def as_tuple(
        self,
    ) -> tuple[
        Position,
        int,
        FrozenSet[Position],
        FrozenSet[Position],
    ]:
        """
        Return the complete observation representation.

        This representation can later be used as a key for
        tabular algorithms.
        """

        return (
            self.position,
            self.battery,
            self.dirty_cells,
            self.dynamic_obstacles,
        )

    def __hash__(self) -> int:
        """Return a hash based on the complete observation."""

        return hash(self.as_tuple())

    def __repr__(self) -> str:
        return (
            "Observation("
            f"position={self.position}, "
            f"battery={self.battery}, "
            f"dirty_cells={self.dirty_cells}, "
            f"dynamic_obstacles={self.dynamic_obstacles}"
            ")"
        )