from __future__ import annotations

from enum import Enum


class Action(Enum):
    """
    Actions available to the cleaning vehicle.

    The value of each action represents the position delta:
        (row_delta, column_delta)

    Grid coordinates:
        - row increases downward
        - column increases to the right
    """

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def delta(self) -> tuple[int, int]:
        """Return the row/column displacement for this action."""
        return self.value