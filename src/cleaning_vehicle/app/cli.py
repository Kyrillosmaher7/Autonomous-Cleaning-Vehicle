"""Command-line entry point: ``python run.py``."""

from __future__ import annotations

import argparse
import sys

from cleaning_vehicle.config import ConfigCatalog, ConfigError

from .builders import build_simulation, prepare_agent


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Autonomous Cleaning Vehicle visualizer. Without arguments "
            "the control panel opens so you can pick a version and model."
        ),
    )
    parser.add_argument("--version", help="version id, e.g. v1 or v2")
    parser.add_argument("--model", help="model id, e.g. q_learning")
    parser.add_argument(
        "--config-dir",
        help="folder with default.yaml, environments/ and agents/ "
             "(default: <project>/configs)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the available versions and models, then exit",
    )
    return parser.parse_args(argv)


def _print_catalog(catalog: ConfigCatalog) -> None:
    print("Versions:")
    for entry in catalog.versions().values():
        print(f"  {entry.id:<12}{entry.name}")

    print("Models:")
    for entry in catalog.models().values():
        print(f"  {entry.id:<12}{entry.name}")


def _run_direct(catalog: ConfigCatalog, version: str, model: str) -> None:
    """Skip the control panel and open the dashboard straight away."""

    from cleaning_vehicle.visualization.dashboard import SimulationDashboard

    scenario = catalog.load(version, model)

    if scenario.agent.requires_training:
        print(
            f"Training {scenario.agent.name} for "
            f"{scenario.training.episodes} episodes..."
        )

    agent = prepare_agent(scenario)

    SimulationDashboard(
        simulation=build_simulation(scenario, agent),
        visualization=scenario.visualization,
        title=f"{scenario.version_name} - {scenario.agent.name}",
        run_details={
            "Version": scenario.version.upper(),
            "Model": scenario.agent.name,
        },
    ).run()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        catalog = ConfigCatalog(args.config_dir)

        if args.list:
            _print_catalog(catalog)
            return 0

        if args.version and args.model:
            _run_direct(catalog, args.version, args.model)
            return 0

        from .launcher import Launcher

        Launcher(catalog, args.version, args.model).run()
        return 0

    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
