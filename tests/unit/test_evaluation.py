from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.training.evaluator import evaluate_agent


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


def test_evaluation_returns_requested_episode_count():
    environment = make_environment()
    agent = QLearningAgent(seed=42)

    result = evaluate_agent(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=10,
        max_steps=20,
    )

    assert result.episodes == 10


def test_evaluation_returns_valid_metrics():
    environment = make_environment()
    agent = QLearningAgent(seed=42)

    result = evaluate_agent(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=10,
        max_steps=20,
    )

    assert result.average_reward >= -1000
    assert result.average_steps >= 0
    assert result.average_cleaned_cells >= 0
    assert 0.0 <= result.completion_rate <= 1.0
    assert result.average_invalid_actions >= 0


def test_evaluation_does_not_change_q_table():
    environment = make_environment()

    agent = QLearningAgent(seed=42)

    # Give the agent some learned values.
    agent.update(
        state=make_state(),
        action=agent.select_action(make_state()),
        reward=10.0,
        next_state=make_state(),
        done=False,
    )

    before = dict(agent.q_table)

    evaluate_agent(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=10,
        max_steps=20,
    )

    assert agent.q_table == before


def test_q_learning_evaluation_restores_epsilon():
    environment = make_environment()

    agent = QLearningAgent(
        epsilon=0.75,
        seed=42,
    )

    original_epsilon = agent.epsilon

    evaluate_agent(
        environment=environment,
        agent=agent,
        initial_state_factory=make_state,
        episodes=10,
        max_steps=20,
    )

    assert agent.epsilon == original_epsilon


def test_evaluation_rejects_invalid_episode_count():
    environment = make_environment()
    agent = QLearningAgent()

    try:
        evaluate_agent(
            environment=environment,
            agent=agent,
            initial_state_factory=make_state,
            episodes=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_evaluation_rejects_invalid_max_steps():
    environment = make_environment()
    agent = QLearningAgent()

    try:
        evaluate_agent(
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