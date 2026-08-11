# System Architecture

## 1. Overview

The Autonomous Cleaning Vehicle project is structured as a modular reinforcement-learning system.

The architecture separates:

* Environment representation
* State representation
* Actions
* Rewards
* Battery management
* Agents
* Episode execution
* Training
* Evaluation
* Experiments

This separation allows individual components to be tested and extended independently.

---

## 2. High-Level Architecture

The V1 system follows this interaction:

```text
                    ┌─────────────────────┐
                    │       State         │
                    │                     │
                    │ position            │
                    │ dirty cells         │
                    │ battery             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Agent         │
                    │                     │
                    │ Random              │
                    │ Greedy              │
                    │ Q-Learning          │
                    └──────────┬──────────┘
                               │
                            Action
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Environment      │
                    │                     │
                    │ movement            │
                    │ boundaries          │
                    │ obstacles           │
                    │ cleaning            │
                    │ battery             │
                    │ charging            │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                Next State              Reward
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Episode Runner    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Evaluation      │
                    └─────────────────────┘
```

---

## 3. Project Structure

The project is organized around the following major components:

```text
src/cleaning_vehicle/
│
├── agents/
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── greedy_agent.py
│   ├── q_learning.py
│   └── sarsa.py
│
├── environment/
│   ├── actions.py
│   ├── battery.py
│   ├── charging_station.py
│   ├── cleaning_env.py
│   ├── grid.py
│   ├── rewards.py
│   └── state.py
│
├── episode.py
├── evaluation.py
│
└── experiments/
    ├── random_baseline.py
    ├── greedy_baseline.py
    └── q_learning_baseline.py
```

The exact implementation may evolve as the project progresses, but the architectural responsibility of each component should remain clear.

---

## 4. Environment Layer

The environment is responsible for representing and enforcing the physical rules of the cleaning world.

### Grid

The `Grid` represents the physical layout.

It manages:

* Width
* Height
* Obstacles
* Dirty cells
* Cell validation
* Traversability
* Cleaning
* Rendering

The grid does not decide which action the vehicle should take.

---

### Cleaning Environment

`CleaningEnvironment` provides the RL interaction interface.

Its primary responsibility is to process:

```text
State + Action
```

and produce:

```text
Next State + Reward + Done
```

The main transition is:

```text
State
  │
  ▼
Action
  │
  ▼
Environment
  │
  ├── Validate movement
  ├── Apply movement
  ├── Clean cell
  ├── Update battery
  ├── Calculate reward
  └── Check completion
  │
  ▼
Next State + Reward + Done
```

---

## 5. State Layer

The `State` represents the dynamic information needed by the RL agent.

V1 state information includes:

```text
State
├── position
├── dirty_cells
└── battery
```

The state is immutable.

This is important because Q-Learning uses states as dictionary keys.

Conceptually:

```python
Q[state][action]
```

can therefore be used to represent the learned Q-values.

Static information such as the obstacle layout is maintained by the environment rather than duplicated inside every state.

---

## 6. Action Layer

The `Action` enum defines the available movement operations:

```text
UP
DOWN
LEFT
RIGHT
```

Each action has a corresponding position delta.

For example:

```text
RIGHT → (0, 1)
```

The action layer contains no environment logic.

It only describes what actions are available.

---

## 7. Reward Layer

The reward system is separated from movement logic.

`RewardConfig` defines configurable reward values.

The V1 default policy is:

```text
Valid movement       → -1
Invalid movement     → -2
Cleaning             → +10
Completion           → +20
```

`RewardCalculator` converts environment events into numerical rewards.

This separation makes it possible to experiment with different reward policies without rewriting the environment.

---

## 8. Battery Layer

Battery management is represented separately from the core grid.

The battery component is responsible for energy-related behavior such as:

* Current battery level
* Maximum battery capacity
* Energy consumption
* Charging
* Detecting insufficient battery

The battery state is exposed to the RL state because available energy can affect future decisions.

The separation also provides a foundation for more sophisticated energy-management behavior in later versions.

---

## 9. Charging Station

The charging station represents a designated position where the vehicle can recharge.

Conceptually:

```text
                 Battery
                    │
                    ▼
             ┌──────────────┐
             │ Vehicle moves│
             └──────┬───────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Charging Station  │
          └─────────┬─────────┘
                    │
                    ▼
              Battery recharge
```

In V1, charging is intentionally simple.

The purpose is to establish the charging mechanism before introducing more complex energy planning.

---

## 10. Agent Layer

Agents are separated from the environment.

An agent receives a state and chooses an action:

```text
State
  │
  ▼
Agent
  │
  ▼
Action
```

The environment does not know how the action was selected.

This allows multiple decision-making strategies to operate on the same environment.

---

## 11. Base Agent

`BaseAgent` defines the common interface for agents.

The central operation is:

```python
select_action(state)
```

Concrete agents implement this interface.

This gives the project a consistent way to run:

* Random Agent
* Greedy Agent
* Q-Learning Agent
* Future agents

---

## 12. Random Agent

The Random Agent provides a simple non-learning baseline.

Its decision process is:

```text
State
  ↓
Random action
  ↓
Environment
```

It does not maintain a learned model or Q-table.

Its purpose is to establish a lower-performance reference point.

---

## 13. Greedy Agent

The Greedy Agent provides a deterministic heuristic baseline.

Its basic strategy is to select an action that moves the vehicle toward a nearby dirty cell.

The Greedy Agent does not learn from previous episodes.

It provides a useful comparison against Q-Learning because it represents a simple manually designed strategy.

---

## 14. Q-Learning Agent

The Q-Learning Agent uses a tabular Q-function.

The Q-table stores estimated values for state-action pairs:

```text
Q(state, action)
```

During training:

```text
State
  ↓
Choose action
  ↓
Environment
  ↓
Next state + reward
  ↓
Q-value update
  ↓
Next iteration
```

The agent uses epsilon-greedy exploration.

---

## 15. Episode Layer

The episode runner coordinates an entire episode.

`run_episode()` connects the agent and environment.

The interaction is:

```text
Initial State
     │
     ▼
┌──────────────┐
│ Select Action│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Environment  │
└──────┬───────┘
       │
       ├── Next State
       ├── Reward
       └── Done
       │
       ▼
    Repeat
```

The episode runner also collects metrics such as:

* Total reward
* Number of steps
* Number of cleaned cells
* Completion status
* Invalid actions

---

## 16. Evaluation Layer

Evaluation is intentionally separated from training.

The evaluator supports two modes:

### Existing Agent

Used for a trained Q-Learning agent:

```text
trained agent
     ↓
evaluation
     ↓
greedy policy
```

### Agent Factory

Used for baseline agents:

```text
agent_factory
     ↓
new agent per episode
     ↓
evaluation
```

This allows Random and Greedy agents to be evaluated independently across many episodes while preserving the learned Q-table for Q-Learning.

---

## 17. Training vs Evaluation

The distinction is important.

### Training

The Q-Learning agent explores and updates its Q-table.

```text
Environment
     ↓
Q-Learning
     ↓
Q-table updates
```

### Evaluation

Learning is disabled.

For Q-Learning:

```text
epsilon = 0
```

The learned policy is then evaluated without exploratory actions.

This ensures that evaluation measures the policy learned during training.

---

## 18. Experiment Layer

Experiments are kept outside the core implementation.

The experiment scripts configure:

* Environment
* Agent
* Training parameters
* Evaluation parameters
* Number of episodes
* Maximum steps
* Output metrics

Current V1 experiments include:

```text
random_baseline.py
greedy_baseline.py
q_learning_baseline.py
```

This keeps experimental configuration separate from reusable library code.

---

## 19. Testing Architecture

The project uses automated unit tests for the individual components.

Testing is organized around component responsibilities.

Examples include:

```text
State tests
Action tests
Grid tests
Environment tests
Reward tests
Battery tests
Agent tests
Episode tests
Evaluation tests
Q-Learning tests
```

The final V1 test suite contains:

```text
214 passed
```

This provides a regression-safety layer before moving to future versions.

---

## 20. Dependency Direction

The architecture follows a simple dependency direction:

```text
Agents
   │
   ▼
State / Actions
   │
   ▼
Environment
   │
   ├── Grid
   ├── Rewards
   ├── Battery
   └── Charging
```

Higher-level orchestration components such as episodes, evaluation, training, and experiments use these lower-level components.

The environment does not depend on a particular learning algorithm.

Therefore the same environment can be used with:

```text
Random
Greedy
Q-Learning
SARSA
Future algorithms
```

---

## 21. Architectural Principle

The most important architectural principle in V1 is:

> **The environment defines what can happen; the agent decides what to do.**

The environment is responsible for enforcing the rules of the world.

The agent is responsible for selecting actions.

The RL algorithm is responsible for learning from experience.

This separation makes the project easier to test, understand, and extend.

---

## 22. Extension Toward V2

The architecture is intentionally designed to support future complexity.

V2 can extend the environment with:

* Dynamic obstacles
* Dirt appearing over time
* More advanced battery management
* Partial observability

The agent interface does not need to fundamentally change.

The environment and state representation can evolve while the same high-level interaction remains:

```text
State
  ↓
Agent
  ↓
Action
  ↓
Environment
  ↓
Next State + Reward + Done
```

This provides the architectural foundation for the next version of the Autonomous Cleaning Vehicle project.
