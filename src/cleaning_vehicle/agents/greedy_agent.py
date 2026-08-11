from __future__ import annotations

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.state import State

from .base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """
    Greedy cleaning agent.

    The agent always targets the nearest dirty cell using
    Manhattan distance and selects a movement that reduces
    that distance.

    This agent does not learn.
    """

    ACTION_PRIORITY = (
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
    )

    @staticmethod
    def _manhattan_distance(
        position: tuple[int, int],
        target: tuple[int, int],
    ) -> int:
        """Return Manhattan distance between two positions."""
        return abs(position[0] - target[0]) + abs(position[1] - target[1])

    def _nearest_dirty_cell(
        self,
        state: State,
    ) -> tuple[int, int] | None:
        """Return the nearest dirty cell."""
        if not state.dirty_cells:
            return None

        return min(
            state.dirty_cells,
            key=lambda cell: (
                self._manhattan_distance(state.position, cell),
                cell[0],
                cell[1],
            ),
        )

    def select_action(self, state: State) -> Action:
        """
        Select an action that moves toward the nearest dirty cell.

        If no dirt remains, return the first deterministic action.
        """

        target = self._nearest_dirty_cell(state)

        if target is None:
            return Action.UP

        row, col = state.position
        target_row, target_col = target

        candidates: list[Action] = []

        if target_row < row:
            candidates.append(Action.UP)
        elif target_row > row:
            candidates.append(Action.DOWN)

        if target_col < col:
            candidates.append(Action.LEFT)
        elif target_col > col:
            candidates.append(Action.RIGHT)

        for action in self.ACTION_PRIORITY:
            if action in candidates:
                return action

        return Action.UP