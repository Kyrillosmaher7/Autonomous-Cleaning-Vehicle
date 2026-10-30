from __future__ import annotations

import random

from .actions import Action
from .observation import Observation
from .rewards import RewardCalculator
from .state import Position, State


class CleaningEnvironment:
    """
    Grid environment for the autonomous cleaning vehicle.

    The environment owns everything that is not part of the agent's
    State: grid size, obstacles, the charging station, dirt spawning,
    dynamic obstacles and the reward policy.

    Transition order inside :meth:`step`:

        move -> battery -> recharge -> clean
             -> dynamic obstacles -> dirt spawn
             -> low-battery penalty -> completion

    Features by version:

        V1: static obstacles, dirt, battery, charging station.
        V2: dynamic obstacles, dirt spawning, partial observation
            (``observation_radius``), reproducible seeding.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        obstacles: frozenset[Position] = frozenset(),
        charging_station: Position = (0, 0),
        max_battery: int = 20,
        reward_calculator: RewardCalculator | None = None,
        low_battery_threshold: int = 5,
        dynamic_obstacles: frozenset[Position] = frozenset(),
        dynamic_obstacle_move_probability: float = 0.0,
        seed: int | None = None,
        dirt_spawn_probability: float = 0.0,
        max_dirty_cells: int | None = None,
        observation_radius: int = 1,
    ) -> None:
        self._validate_parameters(
            rows=rows,
            cols=cols,
            max_battery=max_battery,
            low_battery_threshold=low_battery_threshold,
            dynamic_obstacle_move_probability=dynamic_obstacle_move_probability,
            dirt_spawn_probability=dirt_spawn_probability,
            max_dirty_cells=max_dirty_cells,
            observation_radius=observation_radius,
        )

        self.rows = rows
        self.cols = cols
        self.obstacles = frozenset(obstacles)
        self.dynamic_obstacles = set(dynamic_obstacles)
        self.charging_station = charging_station

        self.max_battery = max_battery
        self.low_battery_threshold = low_battery_threshold
        self.observation_radius = observation_radius

        self.dynamic_obstacle_move_probability = (
            dynamic_obstacle_move_probability
        )
        self.dirt_spawn_probability = dirt_spawn_probability
        self.max_dirty_cells = max_dirty_cells

        self.reward_calculator = reward_calculator or RewardCalculator()

        # Instance-local RNG: never touches Python's global random state.
        self._rng = random.Random(seed)

        self.state: State | None = None

        self._validate_layout()

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_parameters(
        *,
        rows: int,
        cols: int,
        max_battery: int,
        low_battery_threshold: int,
        dynamic_obstacle_move_probability: float,
        dirt_spawn_probability: float,
        max_dirty_cells: int | None,
        observation_radius: int,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        if max_battery <= 0:
            raise ValueError("max_battery must be positive")

        if low_battery_threshold <= 0:
            raise ValueError("low_battery_threshold must be positive")

        if low_battery_threshold > max_battery:
            raise ValueError(
                "low_battery_threshold cannot exceed max_battery"
            )

        if not 0.0 <= dynamic_obstacle_move_probability <= 1.0:
            raise ValueError(
                "dynamic_obstacle_move_probability "
                "must be between 0.0 and 1.0"
            )

        if not 0.0 <= dirt_spawn_probability <= 1.0:
            raise ValueError(
                "dirt_spawn_probability must be between 0.0 and 1.0"
            )

        if max_dirty_cells is not None and max_dirty_cells <= 0:
            raise ValueError(
                "max_dirty_cells must be positive when provided"
            )

        if observation_radius < 0:
            raise ValueError("observation_radius cannot be negative")

    def _validate_layout(self) -> None:
        """Validate the charging station and dynamic obstacles."""

        if not self._is_inside_grid(self.charging_station):
            raise ValueError("charging_station must be inside the grid")

        if self._is_obstacle(self.charging_station):
            raise ValueError("charging_station cannot be an obstacle")

        for position in self.dynamic_obstacles:
            if not self._is_inside_grid(position):
                raise ValueError(
                    f"Dynamic obstacle position {position} "
                    "must be inside the grid."
                )

            if self._is_obstacle(position):
                raise ValueError(
                    "Dynamic obstacles cannot overlap static obstacles."
                )

            if position == self.charging_station:
                raise ValueError(
                    "Dynamic obstacles cannot occupy the charging station."
                )

    # ==========================================================
    # Position helpers
    # ==========================================================

    def _is_inside_grid(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_obstacle(self, position: Position) -> bool:
        """True if position holds a static obstacle."""
        return position in self.obstacles

    def _is_dynamic_obstacle(self, position: Position) -> bool:
        """True if position holds a dynamic obstacle."""
        return position in self.dynamic_obstacles

    def _is_valid_position(self, position: Position) -> bool:
        """True if the vehicle can move onto position."""
        return (
            self._is_inside_grid(position)
            and not self._is_obstacle(position)
            and not self._is_dynamic_obstacle(position)
        )

    @staticmethod
    def _next_position(position: Position, action: Action) -> Position:
        row, col = position
        row_delta, col_delta = action.delta
        return (row + row_delta, col + col_delta)

    # ==========================================================
    # Charging and battery
    # ==========================================================

    def is_charging_station(self, position: Position) -> bool:
        return position == self.charging_station

    def _recharge_if_needed(self, state: State) -> tuple[State, bool]:
        """
        Recharge at the charging station.

        Returns ``(new_state, recharged)``; ``recharged`` is True only
        when the battery actually increased.
        """

        if not self.is_charging_station(state.position):
            return state, False

        if state.battery == self.max_battery:
            return state, False

        return state.recharge(self.max_battery), True

    def is_low_battery(self, state: State) -> bool:
        """True when battery is at or below the low-battery threshold."""
        return state.battery <= self.low_battery_threshold

    def is_critical_battery(self, state: State) -> bool:
        """True when the battery is completely empty."""
        return state.battery == 0

    def has_sufficient_battery(self, state: State) -> bool:
        """True when battery is above the low-battery threshold."""
        return state.battery > self.low_battery_threshold

    def should_return_to_charger(self, state: State) -> bool:
        """
        True when battery is low and the vehicle is away from the charger.

        Only a signal for agents; it does not move the vehicle.
        """
        return (
            self.is_low_battery(state)
            and not self.is_charging_station(state.position)
        )

    # ==========================================================
    # Movement and cleaning
    # ==========================================================

    def move(self, state: State, action: Action) -> State:
        """
        Apply an action without calculating rewards.

        A valid move costs one battery unit. An invalid move (or an
        empty battery) leaves the state unchanged.
        """

        if not isinstance(action, Action):
            raise TypeError("action must be an Action")

        if state.battery == 0:
            return state

        next_position = self._next_position(state.position, action)

        if not self._is_valid_position(next_position):
            return state

        moved = state.move(next_position).consume_battery()
        moved, _ = self._recharge_if_needed(moved)

        return moved

    @staticmethod
    def _clean_current_cell(state: State) -> tuple[State, bool]:
        """Remove dirt from the robot's cell. Returns (state, cleaned)."""

        if state.position not in state.dirty_cells:
            return state, False

        return state.clean(state.position), True

    # ==========================================================
    # Dynamic obstacles
    # ==========================================================

    def _dynamic_obstacle_candidates(
        self,
        position: Position,
        occupied_dynamic_positions: set[Position],
        robot_position: Position,
    ) -> list[Position]:
        """
        Cells a dynamic obstacle may move to.

        Blocked: outside the grid, static obstacles, other dynamic
        obstacles, the robot and the charging station.
        """

        blocked = occupied_dynamic_positions | {
            robot_position,
            self.charging_station,
        }

        candidates: list[Position] = []

        for action in Action:
            candidate = self._next_position(position, action)

            if (
                self._is_inside_grid(candidate)
                and not self._is_obstacle(candidate)
                and candidate not in blocked
            ):
                candidates.append(candidate)

        return candidates

    def _move_dynamic_obstacles(self, robot_position: Position) -> None:
        """
        Move each dynamic obstacle independently with the configured
        probability. Obstacles are processed in sorted order so seeded
        runs stay reproducible.
        """

        if (
            not self.dynamic_obstacles
            or self.dynamic_obstacle_move_probability <= 0.0
        ):
            return

        current_positions = set(self.dynamic_obstacles)
        new_positions: set[Position] = set()

        for position in sorted(current_positions):

            if self._rng.random() > self.dynamic_obstacle_move_probability:
                new_positions.add(position)
                continue

            occupied = new_positions | (current_positions - {position})

            candidates = self._dynamic_obstacle_candidates(
                position=position,
                occupied_dynamic_positions=occupied,
                robot_position=robot_position,
            )

            new_positions.add(
                self._rng.choice(candidates) if candidates else position
            )

        self.dynamic_obstacles = new_positions

    def update_dynamic_obstacles(self, robot_position: Position) -> None:
        """Public API for advancing the dynamic obstacles one tick."""
        self._move_dynamic_obstacles(robot_position=robot_position)

    # ==========================================================
    # Dynamic dirt
    # ==========================================================

    def _dirt_spawn_candidates(self, state: State) -> list[Position]:
        """
        Cells where new dirt may appear.

        Blocked: static/dynamic obstacles, the robot, the charging
        station and cells that are already dirty.
        """

        blocked = (
            set(self.obstacles)
            | self.dynamic_obstacles
            | set(state.dirty_cells)
            | {state.position, self.charging_station}
        )

        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in blocked
        ]

    def _spawn_dirt(self, state: State) -> State:
        """
        Try to spawn one new dirty cell (at most one per step).

        Controlled by ``dirt_spawn_probability`` and capped by
        ``max_dirty_cells``.
        """

        if self.dirt_spawn_probability <= 0.0:
            return state

        if (
            self.max_dirty_cells is not None
            and len(state.dirty_cells) >= self.max_dirty_cells
        ):
            return state

        if self._rng.random() > self.dirt_spawn_probability:
            return state

        candidates = self._dirt_spawn_candidates(state)

        if not candidates:
            return state

        position = self._rng.choice(candidates)

        return State(
            position=state.position,
            dirty_cells=state.dirty_cells | {position},
            battery=state.battery,
        )

    # ==========================================================
    # Partial observability
    # ==========================================================

    def _is_visible(self, origin: Position, target: Position) -> bool:
        """True if target is inside the square observation window."""
        return (
            abs(target[0] - origin[0]) <= self.observation_radius
            and abs(target[1] - origin[1]) <= self.observation_radius
        )

    def get_observation(self, state: State) -> Observation:
        """
        Build the partial observation the agent is allowed to see.

        Always visible: own position and battery.
        Visible only inside the observation radius: dirty cells and
        dynamic obstacles.
        """

        return Observation(
            position=state.position,
            battery=state.battery,
            dirty_cells=frozenset(
                cell
                for cell in state.dirty_cells
                if self._is_visible(state.position, cell)
            ),
            dynamic_obstacles=frozenset(
                obstacle
                for obstacle in self.dynamic_obstacles
                if self._is_visible(state.position, obstacle)
            ),
        )

    # ==========================================================
    # Episode termination
    # ==========================================================

    def _battery_depleted(self, state: State) -> bool:
        return (
            state.battery == 0
            and not self.is_charging_station(state.position)
        )

    def is_done(self, state: State) -> bool:
        """
        True when the episode should terminate.

        - Battery exhaustion away from the charger always terminates.
        - With dirt spawning (V2) an empty dirty set does not
          terminate the episode.
        - Without dirt spawning (V1) the episode ends when all dirt
          is cleaned.
        """

        if self._battery_depleted(state):
            return True

        if self.dirt_spawn_probability > 0.0:
            return False

        return len(state.dirty_cells) == 0

    # ==========================================================
    # RL transition
    # ==========================================================

    def step(
        self,
        state: State,
        action: Action,
    ) -> tuple[State, float, bool]:
        """Execute one transition. Returns (next_state, reward, done)."""

        if not isinstance(action, Action):
            raise TypeError("action must be an Action")

        rewards = self.reward_calculator

        # Empty battery: nothing can happen any more.
        if state.battery == 0:
            return state, rewards.invalid_reward(), True

        next_position = self._next_position(state.position, action)

        if not self._is_valid_position(next_position):
            return state, rewards.invalid_reward(), self.is_done(state)

        # Move (costs one battery unit).
        new_state = state.move(next_position).consume_battery()
        reward = rewards.movement_reward()

        # Recharge.
        new_state, recharged = self._recharge_if_needed(new_state)
        if recharged:
            reward += rewards.charging_reward()

        # Clean.
        new_state, cleaned = self._clean_current_cell(new_state)
        if cleaned:
            reward += rewards.clean_reward()

        # World dynamics.
        self.update_dynamic_obstacles(robot_position=new_state.position)
        new_state = self._spawn_dirt(new_state)

        # Low-battery penalty.
        if (
            new_state.battery <= self.low_battery_threshold
            and not self.is_charging_station(new_state.position)
        ):
            reward += rewards.low_battery_reward()

        # Completion.
        done = self.is_done(new_state)
        if done and cleaned:
            reward += rewards.completion_reward()

        return new_state, reward, done
