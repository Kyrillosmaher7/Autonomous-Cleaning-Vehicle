from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.training.trainer import train_q_learning


def make_state():
    return State(
        position=(0, 0),
        dirty_cells=frozenset({(0, 1)}),
        battery=10,
    )


def make_environment():
    return CleaningEnvironment(
        rows=3,
        cols=3,
    )


def test_training_returns_one_result_per_episode():
    environment = make_environment()
    agent = QLearningAgent(seed=42)

    result = train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=10,
        max_steps=20,
    )

    assert len(result.episode_rewards) == 10
    assert len(result.episode_steps) == 10
    assert len(result.episode_cleaned_cells) == 10
    assert len(result.episode_completions) == 10


def test_training_updates_q_table():
    environment = make_environment()
    agent = QLearningAgent(seed=42)

    assert agent.table_size == 0

    train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=10,
        max_steps=20,
    )

    assert agent.table_size > 0


def test_training_decays_epsilon():
    environment = make_environment()

    agent = QLearningAgent(
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.01,
        seed=42,
    )

    train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=3,
        max_steps=10,
    )

    assert agent.epsilon < 1.0


def test_training_rejects_invalid_episode_count():
    environment = make_environment()
    agent = QLearningAgent()

    try:
        train_q_learning(
            environment=environment,
            agent=agent,
            initial_state_factory=make_state,
            episodes=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_training_rejects_invalid_max_steps():
    environment = make_environment()
    agent = QLearningAgent()

    try:
        train_q_learning(
            environment=environment,
            agent=agent,
            initial_state_factory=make_state,
            episodes=10,
            max_steps=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")