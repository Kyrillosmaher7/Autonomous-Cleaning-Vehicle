from __future__ import annotations

from cleaning_vehicle.environment.actions import Action
from cleaning_vehicle.environment.observation import Observation
from cleaning_vehicle.environment.state import State

from .base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """
    Greedy cleaning agent.

    V1:
        Uses the complete State and therefore knows all
        remaining dirty cells.

    V2:
        Uses only the partial Observation and therefore
        considers only visible dirty cells.

    The agent does not learn.
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

        return (
            abs(position[0] - target[0])
            + abs(position[1] - target[1])
        )

    # ==========================================================
    # V1
    # ==========================================================

    def _nearest_dirty_cell(
        self,
        state: State,
    ) -> tuple[int, int] | None:
        """Return the nearest dirty cell from the complete State."""

        if not state.dirty_cells:
            return None

        return min(
            state.dirty_cells,
            key=lambda cell: (
                self._manhattan_distance(
                    state.position,
                    cell,
                ),
                cell[0],
                cell[1],
            ),
        )

    def select_action(
        self,
        state: State,
    ) -> Action:
        """
        V1 action selection.

        Uses all dirty cells contained in the complete State.
        """

        target = self._nearest_dirty_cell(state)

        if target is None:
            return Action.UP

        return self._move_toward(
            state.position,
            target,
        )

    # ==========================================================
    # V2
    # ==========================================================

    def _nearest_visible_dirty_cell(
        self,
        observation: Observation,
    ) -> tuple[int, int] | None:
        """
        Return the nearest dirty cell visible to the agent.

        Only observation.dirty_cells is considered.
        """

        if not observation.dirty_cells:
            return None

        return min(
            observation.dirty_cells,
            key=lambda cell: (
                self._manhattan_distance(
                    observation.position,
                    cell,
                ),
                cell[0],
                cell[1],
            ),
        )

    def select_observation(
        self,
        observation: Observation,
    ) -> Action:
        """
        V2 action selection.

        The agent can only use information contained in
        the partial observation.

        If no dirty cell is visible, a deterministic default
        action is returned.
        """

        target = self._nearest_visible_dirty_cell(
            observation
        )

        if target is None:
            return Action.UP

        return self._move_toward(
            observation.position,
            target,
        )

    # ==========================================================
    # Shared movement logic
    # ==========================================================

    def _move_toward(
        self,
        position: tuple[int, int],
        target: tuple[int, int],
    ) -> Action:
        """Return an action that reduces distance to target."""

        row, col = position
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