"""
Train (if needed) and evaluate one model on one version, without a GUI.

    python -m cleaning_vehicle.experiments.run_experiment \
        --version v2 --model q_learning

All numbers (grid, rewards, episodes, hyperparameters, seed) come from
the YAML files in ``configs/``.
"""

from __future__ import annotations

import argparse
import sys

from cleaning_vehicle.app.builders import evaluate, prepare_agent
from cleaning_vehicle.config import ConfigCatalog, ConfigError


def run_experiment(
    version: str,
    model: str,
    config_dir: str | None = None,
) -> None:
    scenario = ConfigCatalog(config_dir).load(version, model)

    print(f"{scenario.version_name} | {scenario.agent.name}")
    print("-" * 40)

    agent = prepare_agent(scenario)
    result = evaluate(scenario, agent)

    print(f"Episodes:              {result.episodes}")
    print(f"Average reward:        {result.average_reward:.2f}")
    print(f"Average steps:         {result.average_steps:.2f}")
    print(f"Average cleaned cells: {result.average_cleaned_cells:.2f}")
    print(f"Completion rate:       {result.completion_rate:.2%}")
    print(f"Average invalid moves: {result.average_invalid_actions:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-dir")
    args = parser.parse_args(argv)

    try:
        run_experiment(args.version, args.model, args.config_dir)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
