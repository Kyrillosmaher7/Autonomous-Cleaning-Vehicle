from pathlib import Path

import pytest

from cleaning_vehicle.config import ConfigCatalog, ConfigError, deep_merge
from cleaning_vehicle.config.loader import default_config_dir


@pytest.fixture(scope="module")
def catalog() -> ConfigCatalog:
    return ConfigCatalog()


def test_real_configs_discover_available_versions_and_models(catalog):
    assert set(catalog.versions()) == {"v1", "v2"}
    assert set(catalog.models()) == {"random", "greedy", "q_learning"}


def test_planned_entries_are_hidden(catalog):
    assert "v3" not in catalog.versions()
    assert "sarsa" not in catalog.models()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("model", ["random", "greedy", "q_learning"])
def test_every_available_combination_loads(catalog, version, model):
    scenario = catalog.load(version, model)

    assert scenario.version == version
    assert scenario.agent.algorithm == model


def test_v1_is_full_state_and_v2_is_partial(catalog):
    assert not catalog.load("v1", "greedy").partial_observation
    assert catalog.load("v2", "greedy").partial_observation


def test_version_file_overrides_default_training_steps(catalog):
    assert catalog.load("v1", "greedy").training.max_steps == 100
    assert catalog.load("v2", "greedy").training.max_steps == 200
    assert catalog.load("v2", "greedy").training.episodes == 1000


def test_unknown_version_and_model_are_rejected(catalog):
    with pytest.raises(ConfigError, match="Unknown version"):
        catalog.load("v9", "greedy")

    with pytest.raises(ConfigError, match="Unknown model"):
        catalog.load("v1", "nope")


def test_deep_merge_merges_nested_and_does_not_mutate():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3}, "c": 4}

    merged = deep_merge(base, override)

    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_missing_config_directory_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ConfigCatalog(tmp_path / "missing")


# ----------------------------------------------------------
# Validation, using a temporary copy of the real configs
# ----------------------------------------------------------

@pytest.fixture
def config_copy(tmp_path) -> Path:
    import shutil

    target = tmp_path / "configs"
    shutil.copytree(default_config_dir(), target)
    return target


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not found in {path.name}"
    path.write_text(text.replace(old, new, 1))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("spawn_probability: 0.05", "spawn_probability: 1.5", "<= 1.0"),
        ("rows: 10", "rows: 0", ">= 1"),
        ("max: 50", "max: 3", "cannot exceed max"),
        ("mode: partial", "mode: foggy", "'full' or 'partial'"),
        ("[[1, 1], [3, 3], [5, 5]]", "[[1, 1], [30, 3]]", "outside"),
        ("[[1, 1], [3, 3], [5, 5]]", "[[2, 2]]", "overlaps an obstacle"),
    ],
)
def test_invalid_values_are_rejected(config_copy, old, new, message):
    _edit(config_copy / "environments" / "v2_dynamic.yaml", old, new)

    with pytest.raises(ConfigError, match=message):
        ConfigCatalog(config_copy).load("v2", "greedy")


def test_unknown_key_is_rejected(config_copy):
    _edit(
        config_copy / "environments" / "v2_dynamic.yaml",
        "spawn_probability",
        "spawn_probabilty",
    )

    with pytest.raises(ConfigError, match="Unknown key"):
        ConfigCatalog(config_copy).load("v2", "greedy")


def test_error_names_version_and_model(config_copy):
    _edit(config_copy / "environments" / "v1_static.yaml",
          "rows: 10", "rows: 0")

    with pytest.raises(ConfigError, match="version 'v1', model 'random'"):
        ConfigCatalog(config_copy).load("v1", "random")


def test_new_version_file_appears_without_code_changes(config_copy):
    source = config_copy / "environments" / "v2_dynamic.yaml"
    text = source.read_text().replace("version: v2", "version: v2b")
    (config_copy / "environments" / "v2b.yaml").write_text(text)

    catalog = ConfigCatalog(config_copy)

    assert "v2b" in catalog.versions()
    assert catalog.load("v2b", "greedy").version == "v2b"
