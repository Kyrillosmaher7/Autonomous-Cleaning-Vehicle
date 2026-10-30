"""
Typed, validated configuration objects.

Every numerical value used at runtime (grid size, battery, rewards,
probabilities, hyperparameters, episodes, UI sizes, colors, ...) is
defined in the YAML files under ``configs/`` and parsed into the
dataclasses below. Unknown keys and out-of-range values raise a
:class:`ConfigError` that names the offending setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from cleaning_vehicle.environment.rewards import RewardConfig

Position = tuple[int, int]


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


# ==============================================================
# Parsing helpers
# ==============================================================

def _mapping(data: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{where}' must be a mapping")
    return data


def _check_keys(
    data: Mapping[str, Any],
    where: str,
    allowed: set[str],
) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{where}': {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )


def _number(
    data: Mapping[str, Any],
    key: str,
    where: str,
    *,
    kind: type = float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Any:
    """Read a required number and range-check it."""

    if key not in data:
        raise ConfigError(f"Missing '{where}.{key}'")

    value = data[key]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{where}.{key}' must be a number")

    if kind is int:
        if value != int(value):
            raise ConfigError(f"'{where}.{key}' must be an integer")
        value = int(value)
    else:
        value = float(value)

    if minimum is not None and value < minimum:
        raise ConfigError(f"'{where}.{key}' must be >= {minimum}")

    if maximum is not None and value > maximum:
        raise ConfigError(f"'{where}.{key}' must be <= {maximum}")

    return value


def _position(value: Any, where: str) -> Position:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        )
    ):
        raise ConfigError(f"'{where}' must be a [row, col] pair")

    return (value[0], value[1])


def _positions(value: Any, where: str) -> tuple[Position, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{where}' must be a list of [row, col] pairs")

    return tuple(
        _position(item, f"{where}[{i}]") for i, item in enumerate(value)
    )


# ==============================================================
# Environment sections
# ==============================================================

@dataclass(frozen=True, slots=True)
class GridConfig:
    rows: int
    cols: int

    @classmethod
    def parse(cls, data: Any, where: str) -> "GridConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"rows", "cols"})
        return cls(
            rows=_number(data, "rows", where, kind=int, minimum=1),
            cols=_number(data, "cols", where, kind=int, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class BatteryConfig:
    max: int
    initial: int
    low_threshold: int

    @classmethod
    def parse(cls, data: Any, where: str) -> "BatteryConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"max", "initial", "low_threshold"})
        battery = cls(
            max=_number(data, "max", where, kind=int, minimum=1),
            initial=_number(data, "initial", where, kind=int, minimum=1),
            low_threshold=_number(
                data, "low_threshold", where, kind=int, minimum=1
            ),
        )

        if battery.initial > battery.max:
            raise ConfigError(f"'{where}.initial' cannot exceed max")

        if battery.low_threshold > battery.max:
            raise ConfigError(f"'{where}.low_threshold' cannot exceed max")

        return battery


@dataclass(frozen=True, slots=True)
class DirtConfig:
    initial_cells: tuple[Position, ...]
    spawn_probability: float
    max_dirty_cells: int | None

    @classmethod
    def parse(cls, data: Any, where: str) -> "DirtConfig":
        data = _mapping(data, where)
        _check_keys(
            data,
            where,
            {"initial_cells", "spawn_probability", "max_dirty_cells"},
        )

        max_cells = data.get("max_dirty_cells")
        if max_cells is not None:
            max_cells = _number(
                data, "max_dirty_cells", where, kind=int, minimum=1
            )

        return cls(
            initial_cells=_positions(
                data.get("initial_cells"), f"{where}.initial_cells"
            ),
            spawn_probability=_number(
                data, "spawn_probability", where, minimum=0.0, maximum=1.0
            ),
            max_dirty_cells=max_cells,
        )


@dataclass(frozen=True, slots=True)
class DynamicObstaclesConfig:
    positions: tuple[Position, ...]
    move_probability: float

    @classmethod
    def parse(cls, data: Any, where: str) -> "DynamicObstaclesConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"positions", "move_probability"})
        return cls(
            positions=_positions(
                data.get("positions"), f"{where}.positions"
            ),
            move_probability=_number(
                data, "move_probability", where, minimum=0.0, maximum=1.0
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    mode: str   # "full" or "partial"
    radius: int

    @classmethod
    def parse(cls, data: Any, where: str) -> "ObservationConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"mode", "radius"})

        mode = data.get("mode")
        if mode not in ("full", "partial"):
            raise ConfigError(f"'{where}.mode' must be 'full' or 'partial'")

        return cls(
            mode=mode,
            radius=_number(data, "radius", where, kind=int, minimum=0),
        )

    @property
    def is_partial(self) -> bool:
        return self.mode == "partial"


def _parse_rewards(data: Any, where: str) -> RewardConfig:
    data = _mapping(data, where)
    fields = {
        "clean", "move", "invalid", "completion", "charging", "low_battery"
    }
    _check_keys(data, where, fields)
    return RewardConfig(
        **{name: _number(data, name, where) for name in sorted(fields)}
    )


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    grid: GridConfig
    start_position: Position
    charging_station: Position
    static_obstacles: tuple[Position, ...]
    battery: BatteryConfig
    dirt: DirtConfig
    dynamic_obstacles: DynamicObstaclesConfig
    observation: ObservationConfig
    rewards: RewardConfig

    @classmethod
    def parse(cls, data: Any, where: str = "environment") -> "EnvironmentConfig":
        data = _mapping(data, where)
        _check_keys(
            data,
            where,
            {
                "grid", "start_position", "charging_station",
                "static_obstacles", "battery", "dirt",
                "dynamic_obstacles", "observation", "rewards",
            },
        )

        def section(key: str) -> Any:
            if key not in data:
                raise ConfigError(f"Missing '{where}.{key}'")
            return data[key]

        config = cls(
            grid=GridConfig.parse(section("grid"), f"{where}.grid"),
            start_position=_position(
                section("start_position"), f"{where}.start_position"
            ),
            charging_station=_position(
                section("charging_station"), f"{where}.charging_station"
            ),
            static_obstacles=_positions(
                section("static_obstacles"), f"{where}.static_obstacles"
            ),
            battery=BatteryConfig.parse(
                section("battery"), f"{where}.battery"
            ),
            dirt=DirtConfig.parse(section("dirt"), f"{where}.dirt"),
            dynamic_obstacles=DynamicObstaclesConfig.parse(
                section("dynamic_obstacles"), f"{where}.dynamic_obstacles"
            ),
            observation=ObservationConfig.parse(
                section("observation"), f"{where}.observation"
            ),
            rewards=_parse_rewards(section("rewards"), f"{where}.rewards"),
        )

        config._validate_layout(where)
        return config

    def _validate_layout(self, where: str) -> None:
        """Cross-field checks that need several sections at once."""

        def inside(position: Position) -> bool:
            return (
                0 <= position[0] < self.grid.rows
                and 0 <= position[1] < self.grid.cols
            )

        named = {
            "start_position": [self.start_position],
            "charging_station": [self.charging_station],
            "static_obstacles": self.static_obstacles,
            "dirt.initial_cells": self.dirt.initial_cells,
            "dynamic_obstacles.positions": self.dynamic_obstacles.positions,
        }

        for name, positions in named.items():
            for position in positions:
                if not inside(position):
                    raise ConfigError(
                        f"'{where}.{name}': {list(position)} is outside "
                        f"the {self.grid.rows}x{self.grid.cols} grid"
                    )

        blocked = set(self.static_obstacles) | set(
            self.dynamic_obstacles.positions
        )

        if self.start_position in blocked:
            raise ConfigError(f"'{where}.start_position' is on an obstacle")

        if self.charging_station in blocked:
            raise ConfigError(f"'{where}.charging_station' is on an obstacle")

        if set(self.dirt.initial_cells) & blocked:
            raise ConfigError(
                f"'{where}.dirt.initial_cells' overlaps an obstacle"
            )


# ==============================================================
# Run-level sections
# ==============================================================

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    episodes: int
    max_steps: int

    @classmethod
    def parse(cls, data: Any, where: str = "training") -> "TrainingConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"episodes", "max_steps"})
        return cls(
            episodes=_number(data, "episodes", where, kind=int, minimum=1),
            max_steps=_number(data, "max_steps", where, kind=int, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes: int

    @classmethod
    def parse(cls, data: Any, where: str = "evaluation") -> "EvaluationConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"episodes"})
        return cls(
            episodes=_number(data, "episodes", where, kind=int, minimum=1)
        )


COLOR_KEYS = (
    "empty", "dirty", "static_obstacle", "dynamic_obstacle",
    "robot", "charging", "visible", "grid_line",
)


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    cell_size: int
    speed_default_ms: int
    speed_fastest_ms: int
    speed_slowest_ms: int
    colors: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def parse(
        cls, data: Any, where: str = "visualization"
    ) -> "VisualizationConfig":
        data = _mapping(data, where)
        _check_keys(data, where, {"cell_size", "speed_ms", "colors"})

        speed = _mapping(data.get("speed_ms"), f"{where}.speed_ms")
        _check_keys(speed, f"{where}.speed_ms",
                    {"default", "fastest", "slowest"})
        speed_where = f"{where}.speed_ms"

        fastest = _number(speed, "fastest", speed_where, kind=int, minimum=1)
        slowest = _number(speed, "slowest", speed_where, kind=int, minimum=1)
        default = _number(speed, "default", speed_where, kind=int, minimum=1)

        if not fastest <= default <= slowest:
            raise ConfigError(
                f"'{speed_where}' needs fastest <= default <= slowest"
            )

        colors = _mapping(data.get("colors"), f"{where}.colors")
        _check_keys(colors, f"{where}.colors", set(COLOR_KEYS))

        missing = [key for key in COLOR_KEYS if key not in colors]
        if missing:
            raise ConfigError(f"Missing '{where}.colors': {missing}")

        return cls(
            cell_size=_number(
                data, "cell_size", where, kind=int, minimum=10
            ),
            speed_default_ms=default,
            speed_fastest_ms=fastest,
            speed_slowest_ms=slowest,
            colors=dict(colors),
        )


@dataclass(frozen=True, slots=True)
class AgentConfig:
    algorithm: str
    name: str
    description: str
    requires_training: bool
    hyperparameters: Mapping[str, float]

    @classmethod
    def parse(cls, data: Any, where: str = "agent") -> "AgentConfig":
        data = _mapping(data, where)
        _check_keys(
            data,
            where,
            {
                "algorithm", "name", "description", "status",
                "requires_training", "hyperparameters",
            },
        )

        for key in ("algorithm", "name", "requires_training"):
            if key not in data:
                raise ConfigError(f"Missing '{where}.{key}'")

        if not isinstance(data["requires_training"], bool):
            raise ConfigError(f"'{where}.requires_training' must be true/false")

        hyperparameters = _mapping(
            data.get("hyperparameters", {}), f"{where}.hyperparameters"
        )

        return cls(
            algorithm=str(data["algorithm"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            requires_training=data["requires_training"],
            hyperparameters=dict(hyperparameters),
        )


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Everything needed to build and run one (version, model) pair."""

    version: str
    version_name: str
    seed: int
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    agent: AgentConfig

    @property
    def partial_observation(self) -> bool:
        return self.environment.observation.is_partial

    def summary_lines(self) -> list[str]:
        """Human-readable key values, shown in the launcher."""

        env = self.environment
        lines = [
            f"Grid: {env.grid.rows} x {env.grid.cols}",
            f"Battery: {env.battery.initial} / {env.battery.max} "
            f"(low at {env.battery.low_threshold})",
            f"Dirty cells at start: {len(env.dirt.initial_cells)}",
            f"Static obstacles: {len(env.static_obstacles)}",
            f"Dynamic obstacles: {len(env.dynamic_obstacles.positions)} "
            f"(move p={env.dynamic_obstacles.move_probability})",
            f"Dirt spawn probability: {env.dirt.spawn_probability}",
        ]

        if env.observation.is_partial:
            lines.append(f"Vision: partial, radius {env.observation.radius}")
        else:
            lines.append("Vision: full state")

        lines.append(f"Max steps per episode: {self.training.max_steps}")

        if self.agent.requires_training:
            lines.append(
                f"Training: {self.training.episodes} episodes "
                f"before the run"
            )

        return lines
