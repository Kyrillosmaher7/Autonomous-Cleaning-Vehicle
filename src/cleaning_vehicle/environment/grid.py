
from typing import Iterable


class Grid:
    """
    Represents the physical environment of the cleaning vehicle.

    The grid is a 2D world where each cell can be:
        - Empty / traversable
        - Dirty
        - Obstacle

    This class is responsible only for the environment.
    State, actions, rewards, and learning are handled elsewhere.
    """

    EMPTY = 0
    DIRTY = 1
    OBSTACLE = 2

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Iterable[tuple[int, int]] | None = None,
        dirty_cells: Iterable[tuple[int, int]] | None = None,
    ):
        """
        Create a new grid.

        Parameters
        ----------
        width:
            Number of columns.

        height:
            Number of rows.

        obstacles:
            Optional collection of obstacle positions.

        dirty_cells:
            Optional collection of dirty cell positions.
        """

        if width <= 0:
            raise ValueError("Grid width must be greater than 0.")

        if height <= 0:
            raise ValueError("Grid height must be greater than 0.")

        self.width = width
        self.height = height

        # Store positions as sets for fast lookup.
        self.obstacles = set(obstacles or [])
        self.dirty_cells = set(dirty_cells or [])

        self._validate_positions()


    def _validate_positions(self) -> None:
        """Validate obstacle and dirty-cell positions."""

        for position in self.obstacles:
            if not self.is_valid_position(position):
                raise ValueError(
                    f"Obstacle position {position} is outside the grid."
                )

        for position in self.dirty_cells:
            if not self.is_valid_position(position):
                raise ValueError(
                    f"Dirty cell position {position} is outside the grid."
                )

        overlap = self.obstacles & self.dirty_cells

        if overlap:
            raise ValueError(
                f"A cell cannot be both an obstacle and dirty: {overlap}"
            )


    def is_valid_position(self, position: tuple[int, int]) -> bool:
        """
        Check whether a position exists inside the grid.

        Coordinates use:

            (row, column)

        where:
            row    ∈ [0, height - 1]
            column ∈ [0, width - 1]
        """

        row, column = position

        return (
            0 <= row < self.height
            and 0 <= column < self.width
        )


    def is_obstacle(self, position: tuple[int, int]) -> bool:
        """Return True if the position contains an obstacle."""

        return position in self.obstacles

    def is_dirty(self, position: tuple[int, int]) -> bool:
        """Return True if the position is currently dirty."""

        return position in self.dirty_cells

    def is_traversable(self, position: tuple[int, int]) -> bool:
        """
        Return True if the vehicle can move onto the position.

        A position is traversable when:
            1. It is inside the grid.
            2. It is not an obstacle.
        """

        return (
            self.is_valid_position(position)
            and not self.is_obstacle(position)
        )

    def get_cell(self, position: tuple[int, int]) -> int:
        """
        Return the type of a cell.

        Returns:
            Grid.EMPTY
            Grid.DIRTY
            Grid.OBSTACLE
        """

        if not self.is_valid_position(position):
            raise ValueError(f"Invalid position: {position}")

        if self.is_obstacle(position):
            return self.OBSTACLE

        if self.is_dirty(position):
            return self.DIRTY

        return self.EMPTY



    def clean_cell(self, position: tuple[int, int]) -> bool:
        """
        Clean a dirty cell.

        Returns:
            True  -> cell was dirty and has been cleaned.
            False -> cell was not dirty.
        """

        if not self.is_valid_position(position):
            raise ValueError(f"Invalid position: {position}")

        if position in self.dirty_cells:
            self.dirty_cells.remove(position)
            return True

        return False



    @property
    def total_cells(self) -> int:
        """Return the total number of cells in the grid."""

        return self.width * self.height

    @property
    def total_dirty_cells(self) -> int:
        """Return the number of currently dirty cells."""

        return len(self.dirty_cells)

    @property
    def total_obstacles(self) -> int:
        """Return the number of obstacle cells."""

        return len(self.obstacles)

    @property
    def all_clean(self) -> bool:
        """Return True when there are no dirty cells remaining."""

        return len(self.dirty_cells) == 0

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(
        self,
        obstacles: Iterable[tuple[int, int]] | None = None,
        dirty_cells: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """
        Reset the environment configuration.

        If new obstacles or dirty cells are provided,
        they replace the current configuration.
        """

        if obstacles is not None:
            self.obstacles = set(obstacles)

        if dirty_cells is not None:
            self.dirty_cells = set(dirty_cells)

        self._validate_positions()



    def render(self) -> None:
        """
        Print the grid in a human-readable form.

        Symbols:

            . = Empty
            D = Dirty
            # = Obstacle
        """

        for row in range(self.height):
            cells = []

            for column in range(self.width):
                position = (row, column)

                if self.is_obstacle(position):
                    cells.append("#")

                elif self.is_dirty(position):
                    cells.append("D")

                else:
                    cells.append(".")

            print(" ".join(cells))

        print()


    def __repr__(self) -> str:
        return (
            f"Grid("
            f"width={self.width}, "
            f"height={self.height}, "
            f"obstacles={len(self.obstacles)}, "
            f"dirty_cells={len(self.dirty_cells)}"
            f")"
        )

