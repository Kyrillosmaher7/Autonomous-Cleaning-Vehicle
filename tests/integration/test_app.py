import os
from dataclasses import replace

import pytest

from cleaning_vehicle.app.builders import (
    build_agent,
    build_environment,
    build_simulation,
    evaluate,
    prepare_agent,
)
from cleaning_vehicle.config import ConfigCatalog


@pytest.fixture(scope="module")
def catalog() -> ConfigCatalog:
    return ConfigCatalog()


def _fast(scenario):
    """Same scenario with a short training run."""
    return replace(
        scenario, training=replace(scenario.training, episodes=20)
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("model", ["random", "greedy", "q_learning"])
def test_every_combination_runs_to_the_end(catalog, version, model):
    scenario = _fast(catalog.load(version, model))

    agent = prepare_agent(scenario)
    simulation = build_simulation(scenario, agent)

    while simulation.step():
        pass

    assert simulation.finished
    assert simulation.step_count <= scenario.training.max_steps


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_evaluate_uses_config_episode_count(catalog, version):
    scenario = replace(
        _fast(catalog.load(version, "greedy")),
        evaluation=replace(
            catalog.load(version, "greedy").evaluation, episodes=3
        ),
    )

    result = evaluate(scenario, prepare_agent(scenario))

    assert result.episodes == 3


def test_environment_reflects_config_values(catalog):
    scenario = catalog.load("v2", "greedy")
    env = build_environment(scenario)

    assert (env.rows, env.cols) == (10, 10)
    assert env.max_battery == scenario.environment.battery.max
    assert env.observation_radius == scenario.environment.observation.radius
    assert env.dirt_spawn_probability == 0.05
    assert env.reward_calculator.config == scenario.environment.rewards
    assert len(env.obstacles) == 3
    assert len(env.dynamic_obstacles) == 2


def test_q_learning_hyperparameters_come_from_config(catalog):
    agent = build_agent(catalog.load("v1", "q_learning"))

    assert agent.learning_rate == 0.1
    assert agent.discount_factor == 0.95
    assert agent.epsilon_decay == 0.995


def test_trained_q_learning_agent_is_greedy_for_the_run(catalog):
    agent = prepare_agent(_fast(catalog.load("v1", "q_learning")))

    assert agent.epsilon == 0.0


def test_each_build_gives_an_independent_environment(catalog):
    scenario = catalog.load("v2", "greedy")

    assert build_environment(scenario) is not build_environment(scenario)


# ----------------------------------------------------------
# GUI (needs a display; skipped otherwise, e.g. plain CI)
# ----------------------------------------------------------

def _display_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:
        return False


needs_display = pytest.mark.skipif(
    not _display_available(), reason="no display / tkinter available"
)


@needs_display
def test_launcher_lists_choices_and_opens_a_dashboard(catalog):
    from cleaning_vehicle.app.launcher import Launcher

    launcher = Launcher(catalog, version="v2", model="greedy")

    try:
        assert launcher.selected_version == "v2"
        assert launcher.selected_model == "greedy"
        assert len(launcher.version_box["values"]) == 2
        assert "partial" in launcher.summary.get()

        scenario = catalog.load("v2", "greedy")
        dashboard = launcher._open_dashboard(
            scenario, prepare_agent(scenario)
        )
        dashboard.step_once()
        dashboard.reset()

        assert dashboard.simulation.step_count == 0

        dashboard.window.destroy()
    finally:
        launcher.root.destroy()
