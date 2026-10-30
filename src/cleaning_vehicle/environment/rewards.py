from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Reward values used by the environment.

    The values come from the ``rewards`` section of the environment
    YAML file at runtime. The defaults here only apply when the
    environment is built directly in code (for example in tests).
    """

    clean: float = 10.0
    move: float = -1.0
    invalid: float = -2.0
    completion: float = 20.0
    charging: float = 1.0
    low_battery: float = -1.0


class RewardCalculator:
    """Thin accessor around a :class:`RewardConfig`."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def movement_reward(self) -> float:
        """Reward for a valid movement."""
        return self.config.move

    def invalid_reward(self) -> float:
        """Penalty for an invalid movement."""
        return self.config.invalid

    def clean_reward(self) -> float:
        """Reward for cleaning a dirty cell."""
        return self.config.clean

    def completion_reward(self) -> float:
        """Bonus for cleaning the last dirty cell."""
        return self.config.completion

    def charging_reward(self) -> float:
        """Reward for actually recharging the battery."""
        return self.config.charging

    def low_battery_reward(self) -> float:
        """Penalty applied while operating at low battery."""
        return self.config.low_battery
