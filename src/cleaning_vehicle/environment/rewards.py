from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Reward configuration for V1.

    The values are intentionally kept configurable so the
    reward policy can be changed without modifying the
    environment implementation.
    """

    clean: float = 10.0
    move: float = -1.0
    invalid: float = -2.0
    completion: float = 20.0


class RewardCalculator:
    """
    Calculates rewards for the cleaning vehicle.

    Reward policy:

        Clean dirty cell   -> +10
        Valid movement     -> -1
        Invalid movement   -> -2
        Complete cleaning  -> +20
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
    ) -> None:
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
        """Bonus for completing the cleaning task."""
        return self.config.completion