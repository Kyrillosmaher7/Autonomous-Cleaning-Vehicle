from __future__ import annotations

from cleaning_vehicle.agents.q_learning import QLearningAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.training.evaluator import evaluate_agent
from cleaning_vehicle.training.trainer import train_q_learning


ROWS = 10
COLS = 10

START_POSITION = (0, 0)

DIRTY_CELLS = frozenset({
    (1, 1),
    (3, 3),
    (5, 5),
})

BATTERY = 50

TRAINING_EPISODES = 1000
EVALUATION_EPISODES = 100

MAX_STEPS = 100


def create_environment() -> CleaningEnvironment:
    return CleaningEnvironment(
        rows=ROWS,
        cols=COLS,
    )


def create_state() -> State:
    return State(
        position=START_POSITION,
        dirty_cells=DIRTY_CELLS,
        battery=BATTERY,
    )


def main() -> None:
    environment = create_environment()

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )

    # ==============================================================
    # Training
    # ==============================================================

    training = train_q_learning(
        environment=environment,
        agent=agent,
        initial_state_factory=create_state,
        episodes=TRAINING_EPISODES,
        max_steps=MAX_STEPS,
    )

    print("Q-Learning Training")
    print("===================")
    print(f"Episodes:       {TRAINING_EPISODES}")
    print(f"Final epsilon:  {agent.epsilon:.4f}")
    print(f"Q-table size:   {agent.table_size}")

    # ==============================================================
    # Evaluation
    # ==============================================================

    evaluation = evaluate_agent(
        environment=environment,
        agent=agent,
        initial_state_factory=create_state,
        episodes=EVALUATION_EPISODES,
        max_steps=MAX_STEPS,
    )

    print()
    print("Q-Learning Evaluation")
    print("=====================")
    print(
        f"Episodes:              {evaluation.episodes}"
    )
    print(
        f"Average reward:        "
        f"{evaluation.average_reward:.2f}"
    )
    print(
        f"Average steps:         "
        f"{evaluation.average_steps:.2f}"
    )
    print(
        f"Average cleaned cells: "
        f"{evaluation.average_cleaned_cells:.2f}"
    )
    print(
        f"Completion rate:       "
        f"{100 * evaluation.completion_rate:.2f}%"
    )
    print(
        f"Average invalid moves: "
        f"{evaluation.average_invalid_actions:.2f}"
    )


if __name__ == "__main__":
    main()