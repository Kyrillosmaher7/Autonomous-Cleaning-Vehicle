from __future__ import annotations

from .actions import Action
from .rewards import RewardCalculator
from .state import Position, State


class CleaningEnvironment:
    """
    Environment for the autonomous cleaning vehicle.

    Responsibilities:
        - Maintain grid dimensions.
        - Maintain static obstacles.
        - Maintain charging station.
        - Apply actions.
        - Prevent invalid movement.
        - Consume battery.
        - Recharge at charging station.
        - Clean dirty cells.
        - Calculate rewards.
        - Detect episode completion.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        obstacles: frozenset[Position] = frozenset(),
        charging_station: Position = (0, 0),
        max_battery: int = 20,
        reward_calculator: RewardCalculator | None = None,
    ) -> None:

        if rows <= 0 or cols <= 0:
            raise ValueError(
                "rows and cols must be positive"
            )

        if max_battery <= 0:
            raise ValueError(
                "max_battery must be positive"
            )

        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles
        self.charging_station = charging_station
        self.max_battery = max_battery

        self.reward_calculator = (
            reward_calculator or RewardCalculator()
        )

        self.state: State | None = None

        # Validate static environment configuration.
        if not self._is_inside_grid(charging_station):
            raise ValueError(
                "charging_station must be inside the grid"
            )

        if self._is_obstacle(charging_station):
            raise ValueError(
                "charging_station cannot be an obstacle"
            )

    # ==========================================================
    # Position validation
    # ==========================================================

    def _is_inside_grid(
        self,
        position: Position,
    ) -> bool:
        """Return True if position is inside the grid."""

        row, col = position

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )

    def _is_obstacle(
        self,
        position: Position,
    ) -> bool:
        """Return True if position contains an obstacle."""

        return position in self.obstacles

    def _is_valid_position(
        self,
        position: Position,
    ) -> bool:
        """Return True if position is traversable."""

        return (
            self._is_inside_grid(position)
            and not self._is_obstacle(position)
        )

    # ==========================================================
    # Charging
    # ==========================================================

    def is_charging_station(
        self,
        position: Position,
    ) -> bool:
        """Return True if position is the charging station."""

        return position == self.charging_station

    def _recharge_if_needed(
        self,
        state: State,
    ) -> State:
        """
        Recharge the vehicle when it reaches the
        charging station.
        """

        if not self.is_charging_station(state.position):
            return state

        if state.battery == self.max_battery:
            return state

        return state.recharge(self.max_battery)

    # ==========================================================
    # Movement
    # ==========================================================

    def _next_position(
        self,
        position: Position,
        action: Action,
    ) -> Position:
        """Calculate the position resulting from an action."""

        row, col = position
        row_delta, col_delta = action.delta

        return (
            row + row_delta,
            col + col_delta,
        )

    def move(
        self,
        state: State,
        action: Action,
    ) -> State:
        """
        Apply an action without calculating rewards.

        Valid movement consumes one battery unit.

        Invalid movement leaves the state unchanged.
        """

        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action"
            )

        # No movement is possible with an empty battery.
        if state.battery == 0:
            return state

        next_position = self._next_position(
            state.position,
            action,
        )

        if not self._is_valid_position(next_position):
            return state

        new_state = State(
            position=next_position,
            dirty_cells=state.dirty_cells,
            battery=state.battery - 1,
        )

        return self._recharge_if_needed(new_state)

    # ==========================================================
    # Cleaning
    # ==========================================================

    def _clean_current_cell(
        self,
        state: State,
    ) -> tuple[State, bool]:
        """
        Remove dirt from the robot's current cell.

        Returns:
            (new_state, was_cleaned)
        """

        if state.position not in state.dirty_cells:
            return state, False

        remaining_dirty_cells = frozenset(
            cell
            for cell in state.dirty_cells
            if cell != state.position
        )

        new_state = State(
            position=state.position,
            dirty_cells=remaining_dirty_cells,
            battery=state.battery,
        )

        return new_state, True

    # ==========================================================
    # Episode termination
    # ==========================================================

    def is_done(
        self,
        state: State,
    ) -> bool:
        """
        Return True when:

            1. All dirt has been cleaned, or
            2. Battery is empty away from charger.
        """

        if len(state.dirty_cells) == 0:
            return True

        if (
            state.battery == 0
            and not self.is_charging_station(state.position)
        ):
            return True

        return False

    # ==========================================================
    # RL transition
    # ==========================================================

    def step(
        self,
        state: State,
        action: Action,
    ) -> tuple[State, float, bool]:
        """
        Execute one environment transition.

        Transition:

            State + Action
                ↓
            Environment
                ↓
            Next State + Reward + Done
        """

        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action"
            )

        # ------------------------------------------------------
        # Terminal state
        # ------------------------------------------------------

        if self.is_done(state):
            return (
                state,
                0.0,
                True,
            )

        # ------------------------------------------------------
        # Empty battery
        # ------------------------------------------------------

        if state.battery == 0:
            return (
                state,
                self.reward_calculator.invalid_reward(),
                True,
            )

        # ------------------------------------------------------
        # Calculate next position
        # ------------------------------------------------------

        next_position = self._next_position(
            state.position,
            action,
        )

        # ------------------------------------------------------
        # Invalid movement
        # ------------------------------------------------------

        if not self._is_valid_position(next_position):
            return (
                state,
                self.reward_calculator.invalid_reward(),
                self.is_done(state),
            )

        # ------------------------------------------------------
        # Valid movement + battery consumption
        # ------------------------------------------------------

        moved_state = State(
            position=next_position,
            dirty_cells=state.dirty_cells,
            battery=state.battery - 1,
        )

        reward = self.reward_calculator.movement_reward()

        # ------------------------------------------------------
        # Recharge if we reached the charger
        # ------------------------------------------------------

        moved_state = self._recharge_if_needed(
            moved_state
        )

        # ------------------------------------------------------
        # Clean dirt
        # ------------------------------------------------------

        cleaned_state, was_cleaned = (
            self._clean_current_cell(moved_state)
        )

        if was_cleaned:
            reward += (
                self.reward_calculator.clean_reward()
            )

        # ------------------------------------------------------
        # Completion
        # ------------------------------------------------------

        done = self.is_done(cleaned_state)

        if done and was_cleaned:
            reward += (
                self.reward_calculator.completion_reward()
            )

        return (
            cleaned_state,
            reward,
            done,
        )