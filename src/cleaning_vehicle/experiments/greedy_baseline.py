from cleaning_vehicle.agents import GreedyAgent
from cleaning_vehicle.environment.cleaning_env import CleaningEnvironment
from cleaning_vehicle.environment.state import State
from cleaning_vehicle.training.evaluator import evaluate_agent


def create_state() -> State:
    return State(
        position=(0, 0),
        dirty_cells=frozenset({
            (2, 3),
            (5, 5),
            (7, 8),
        }),
    )


def main() -> None:
    environment = CleaningEnvironment(
        rows=10,
        cols=10,
    )

    result = evaluate_agent(
        environment=environment,
        agent_factory=GreedyAgent,
        initial_state_factory=create_state,
        episodes=100,
        max_steps=1000,
    )

    print("Greedy Agent Baseline")
    print("=====================")
    print(f"Episodes:              {result.episodes}")
    print(f"Average reward:        {result.average_reward:.2f}")
    print(f"Average steps:         {result.average_steps:.2f}")
    print(
        "Average cleaned cells: "
        f"{result.average_cleaned_cells:.2f}"
    )
    print(
        "Completion rate:       "
        f"{result.completion_rate * 100:.2f}%"
    )
    print(
        "Average invalid moves: "
        f"{result.average_invalid_actions:.2f}"
    )


if __name__ == "__main__":
    main()