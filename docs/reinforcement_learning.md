# Reinforcement Learning Formulation

## Overview

The Autonomous Cleaning Vehicle is modeled as a Markov Decision Process (MDP).

The agent interacts with a grid-based environment by observing its current state, selecting an action, receiving a reward, and transitioning to a new state.

The V1 formulation is:

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
  ↓
Agent
```

---

# 1. Markov Decision Process

The V1 problem can be represented as:

```text
MDP = (S, A, P, R, γ)
```

where:

* `S` = state space
* `A` = action space
* `P` = transition dynamics
* `R` = reward function
* `γ` = discount factor

---

# 2. State Space

The V1 state contains:

```text
State = (
    robot_position,
    remaining_dirty_cells
)
```

The robot position is represented as:

```text
(row, column)
```

The remaining dirty cells are represented as a `frozenset`.

Example:

```text
State(
    position=(2, 2),
    dirty_cells={
        (1, 3),
        (4, 4),
        (0, 0)
    }
)
```

The state is immutable and hashable, making it suitable for a Q-table.

---

# 3. Action Space

The V1 action space contains four movement actions:

```text
A = {
    UP,
    DOWN,
    LEFT,
    RIGHT
}
```

Their position changes are:

```text
UP     = (-1,  0)
DOWN   = ( 1,  0)
LEFT   = ( 0, -1)
RIGHT  = ( 0,  1)
```

The vehicle does not have a dedicated `CLEAN` action.

Cleaning occurs automatically when the vehicle enters a dirty cell.

---

# 4. Transition Function

The environment determines the next state after an action.

For a valid action:

```text
Current position
       ↓
Apply action delta
       ↓
New position
       ↓
Check dirty cell
       ↓
Clean if necessary
       ↓
Next state
```

For an invalid action:

```text
Current position
       ↓
Invalid target
       ↓
Position unchanged
       ↓
Invalid-action penalty
```

Invalid actions include attempting to:

* Leave the grid
* Enter an obstacle

---

# 5. Reward Function

The V1 reward function is:

```text
Valid movement       = -1
Invalid movement     = -2
Cleaning dirt        = +10
Completion           = +20
```

Therefore, the agent is encouraged to:

```text
Reach dirt quickly
        ↓
Avoid invalid actions
        ↓
Clean dirty cells
        ↓
Complete the task
```

See [Reward Function](reward_function.md) for the complete reward specification.

---

# 6. Episode Termination

An episode terminates when:

```text
remaining_dirty_cells == 0
```

Therefore:

```text
All dirty cells cleaned
          ↓
        done=True
```

A maximum number of steps can also be used during experiments to prevent an agent from running indefinitely.

---

# 7. Q-Learning

Q-Learning estimates the value of each state-action pair:

```text
Q(s, a)
```

The objective is to learn which action produces the highest expected long-term reward.

The update rule is:

```text
Q(s,a) ← Q(s,a)
         + α [r + γ max Q(s',a') - Q(s,a)]
```

where:

```text
s   = current state
a   = selected action
r   = received reward
s'  = next state
a'  = possible next action
α   = learning rate
γ   = discount factor
```

---

# 8. Exploration vs Exploitation

During training, the Q-Learning agent uses epsilon-greedy selection.

### Exploration

With probability `ε`:

```text
Choose a random action
```

This allows the agent to discover new behaviors.

### Exploitation

With probability:

```text
1 - ε
```

the agent selects the action with the highest known Q-value.

---

# 9. Training Process

The V1 training loop is:

```text
Initialize Q-table
       ↓
Initialize environment
       ↓
Create initial state
       ↓
Select action using ε-greedy
       ↓
Environment.step(action)
       ↓
Receive next state + reward
       ↓
Update Q(s,a)
       ↓
Repeat
       ↓
Decay epsilon
       ↓
Next episode
```

The current experiment trains for:

```text
1000 episodes
```

and reaches:

```text
Final epsilon = 0.0100
```

---

# 10. Evaluation Process

Training and evaluation are separated.

During evaluation:

```text
epsilon = 0
```

The agent therefore uses the learned policy without exploration.

Current evaluation:

```text
Average reward:        40.00
Average steps:         10.00
Average cleaned cells:  3.00
Completion rate:      100.00%
Invalid actions:       0.00
```

---

# 11. Baselines

Reinforcement learning performance is evaluated against simpler strategies.

### Random

```text
No learning
Random actions
```

### Greedy

```text
No learning
Move toward nearest dirty cell
```

### Q-Learning

```text
Learns from experience
Uses Q-table
Uses reward feedback
```

This comparison helps determine whether learning provides a meaningful advantage.

---

# 12. V1 MDP Summary

| Component           | V1                               |
| ------------------- | -------------------------------- |
| State               | Position + remaining dirty cells |
| Actions             | UP, DOWN, LEFT, RIGHT            |
| Environment         | Static grid                      |
| Obstacles           | Static                           |
| Dirt                | Initial/static                   |
| Reward              | Movement + cleaning + completion |
| Learning algorithm  | Q-Learning                       |
| Q representation    | Tabular                          |
| Exploration         | Epsilon-greedy                   |
| Episode termination | All dirt cleaned                 |
| Evaluation          | Greedy policy                    |

---

# 13. Current V1 Learning Result

The trained Q-Learning agent currently achieves:

```text
Training episodes:     1000
Final epsilon:         0.0100
Q-table entries:       10816

Evaluation episodes:   100
Average reward:         40.00
Average steps:          10.00
Average cleaned cells:   3.00
Completion rate:       100.00%
Invalid actions:         0.00
```

This confirms that the V1 MDP formulation is learnable by the implemented tabular Q-Learning agent.

---

# 14. Future Extensions

The V1 formulation intentionally provides a simple foundation.

Future versions can extend the MDP with additional state information and environmental dynamics.

Potential extensions include:

```text
V2
├── Dynamic obstacles
├── Dirt appearing over time
├── More complex battery management
└── Partial observability

V3
├── Random baseline
├── Greedy baseline
├── Q-Learning
└── SARSA comparison

V4
├── Deep Q-Network
├── Experience Replay
└── Target Network
```

The V1 implementation therefore establishes the basic RL pipeline before introducing additional complexity.
