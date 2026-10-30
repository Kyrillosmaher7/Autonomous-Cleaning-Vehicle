from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.actions import Action
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


from cleaning_vehicle.environment.observation import Observation


def test_train_q_learning_can_train_from_observations():
    """
    V2:
    Q-Learning training can use partial observations
    without changing the V1 State-based training API.
    """
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        observation_radius=1,
    )

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(1, 1),
            battery=10,
            dirty_cells=frozenset({
                (1, 2),
            }),
        )

    result = train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=1,
        max_steps=10,
        use_observations=True,
    )

    assert result.episode_rewards
    assert result.episode_steps
    assert result.episode_cleaned_cells
    assert result.episode_completions


def test_train_q_learning_observation_mode_uses_observation_q_table():
    """
    V2:
    Observation-based training must populate the
    observation Q-table rather than the V1 State Q-table.
    """
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        observation_radius=1,
    )

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(1, 1),
            battery=10,
            dirty_cells=frozenset({
                (1, 2),
            }),
        )

    train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=1,
        max_steps=5,
        use_observations=True,
    )

    assert agent.observation_table_size > 0

def test_v2_training_updates_only_observation_q_table():
    """
    V2:
    Observation-based training must update the observation
    Q-table and leave the V1 State Q-table untouched.
    """
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        observation_radius=1,
    )

    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(1, 1),
            battery=10,
            dirty_cells=frozenset({
                (1, 2),
            }),
        )

    assert agent.table_size == 0
    assert agent.observation_table_size == 0

    train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=1,
        max_steps=1,
        use_observations=True,
    )

    assert agent.observation_table_size > 0
    assert agent.table_size == 0


def test_v2_training_decays_epsilon():
    """
    V2:
    Observation-based training must use the same epsilon
    decay mechanism as V1.
    """
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        observation_radius=1,
    )

    agent = QLearningAgent(
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.01,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(1, 1),
            battery=10,
            dirty_cells=frozenset({
                (1, 2),
            }),
        )

    train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=1,
        max_steps=1,
        use_observations=True,
    )

    assert agent.epsilon == 0.5


def test_v2_training_respects_max_steps():
    """
    V2:
    Training must stop when max_steps is reached even when
    the cleaning task has not completed.
    """
    environment = CleaningEnvironment(
        rows=5,
        cols=5,
        observation_radius=1,
    )

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(0, 0),
            battery=20,
            dirty_cells=frozenset({
                (4, 4),
            }),
        )

    result = train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=1,
        max_steps=3,
        use_observations=True,
    )

    assert result.episode_steps == (3,)
    assert result.episode_completions == (False,)


def test_v2_training_returns_same_result_structure_as_v1():
    """
    V2:
    Observation-based training must return the same
    TrainingResult structure used by V1.
    """
    environment = CleaningEnvironment(
        rows=3,
        cols=3,
        observation_radius=1,
    )

    agent = QLearningAgent(
        epsilon=0.0,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(1, 1),
            battery=10,
            dirty_cells=frozenset({
                (1, 2),
            }),
        )

    result = train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=3,
        max_steps=10,
        use_observations=True,
    )

    assert len(result.episode_rewards) == 3
    assert len(result.episode_steps) == 3
    assert len(result.episode_cleaned_cells) == 3
    assert len(result.episode_completions) == 3


def test_v2_training_handles_terminal_transition():
    """
    V2:
    When the next state is terminal, the update must use
    only the immediate reward and must not bootstrap from
    the next observation.
    """
    environment = CleaningEnvironment(
        rows=1,
        cols=2,
        observation_radius=1,
    )

    agent = QLearningAgent(
        learning_rate=1.0,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    def initial_state_factory() -> State:
        return State(
            position=(0, 0),
            battery=10,
            dirty_cells=frozenset({
                (0, 1),
            }),
        )

    # Seed the V2 Q-table so the greedy agent chooses RIGHT.
    observation = environment.get_observation(
        initial_state_factory()
    )

    agent.observation_q_table[
        (observation, Action.RIGHT)
    ] = 1.0

    result = train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=initial_state_factory,
        episodes=1,
        max_steps=10,
        use_observations=True,
    )

    assert result.episode_completions == (True,)
    assert result.episode_steps == (1,)

    learned_value = agent.get_observation_q_value(
        observation,
        Action.RIGHT,
    )

    # Cleaning reward is 29 and terminal transition
    # must not include a future Q-value.
    assert learned_value == 29.0