"""Discover and load the YAML configuration files."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    AgentConfig,
    ConfigError,
    EnvironmentConfig,
    EvaluationConfig,
    ScenarioConfig,
    TrainingConfig,
    VisualizationConfig,
)

CONFIG_DIR_ENV_VAR = "CLEANING_VEHICLE_CONFIG_DIR"

# src/cleaning_vehicle/config/loader.py -> project root is 4 levels up
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ENVIRONMENT_KEYS = {
    "version", "name", "description", "status",
    "environment", "training", "evaluation", "visualization", "seed",
}


def default_config_dir() -> Path:
    """``$CLEANING_VEHICLE_CONFIG_DIR`` or ``<project>/configs``."""

    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    return Path(override) if override else _PROJECT_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file that must contain a mapping."""

    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at top level")

    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated by ``override``; nested dicts merge."""

    merged = copy.deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A selectable version or model, as shown in the launcher."""

    id: str
    name: str
    description: str
    path: Path


class ConfigCatalog:
    """
    Finds the versions (``environments/*.yaml``) and models
    (``agents/*.yaml``) available in a config directory and builds a
    validated :class:`ScenarioConfig` for any (version, model) pair.
    """

    def __init__(self, config_dir: Path | str | None = None) -> None:
        self.config_dir = (
            Path(config_dir) if config_dir else default_config_dir()
        )

        if not self.config_dir.is_dir():
            raise ConfigError(
                f"Config directory not found: {self.config_dir}"
            )

    # ----------------------------------------------------------
    # Discovery
    # ----------------------------------------------------------

    def _discover(
        self, folder: str, id_key: str
    ) -> dict[str, CatalogEntry]:
        entries: dict[str, CatalogEntry] = {}

        for path in sorted((self.config_dir / folder).glob("*.yaml")):
            data = load_yaml(path)

            if data.get("status", "available") != "available":
                continue

            if id_key not in data:
                raise ConfigError(f"{path} is missing '{id_key}'")

            entry_id = str(data[id_key])

            if entry_id in entries:
                raise ConfigError(
                    f"Duplicate {id_key} '{entry_id}' in {folder}/"
                )

            entries[entry_id] = CatalogEntry(
                id=entry_id,
                name=str(data.get("name", entry_id)),
                description=str(data.get("description", "")),
                path=path,
            )

        return entries

    def versions(self) -> dict[str, CatalogEntry]:
        """Available environment versions, keyed by version id."""
        return self._discover("environments", "version")

    def models(self) -> dict[str, CatalogEntry]:
        """Available agents/models, keyed by algorithm id."""
        return self._discover("agents", "algorithm")

    # ----------------------------------------------------------
    # Loading
    # ----------------------------------------------------------

    def load(self, version: str, model: str) -> ScenarioConfig:
        """Load, merge and validate the config for (version, model)."""

        versions = self.versions()
        models = self.models()

        if version not in versions:
            raise ConfigError(
                f"Unknown version '{version}'. Available: {sorted(versions)}"
            )

        if model not in models:
            raise ConfigError(
                f"Unknown model '{model}'. Available: {sorted(models)}"
            )

        defaults = load_yaml(self.config_dir / "default.yaml")
        version_data = load_yaml(versions[version].path)

        unknown = set(version_data) - _ENVIRONMENT_KEYS
        if unknown:
            raise ConfigError(
                f"Unknown key(s) in {versions[version].path.name}: "
                f"{sorted(unknown)}"
            )

        merged = deep_merge(defaults, version_data)
        agent_data = load_yaml(models[model].path)

        try:
            seed = merged["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ConfigError("'seed' must be an integer")

            return ScenarioConfig(
                version=version,
                version_name=versions[version].name,
                seed=seed,
                environment=EnvironmentConfig.parse(
                    merged.get("environment")
                ),
                training=TrainingConfig.parse(merged.get("training")),
                evaluation=EvaluationConfig.parse(merged.get("evaluation")),
                visualization=VisualizationConfig.parse(
                    merged.get("visualization")
                ),
                agent=AgentConfig.parse(agent_data),
            )
        except ConfigError as error:
            raise ConfigError(
                f"{error} (version '{version}', model '{model}')"
            ) from error
