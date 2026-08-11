# Algorithms

## Overview

Version 1 implements three agent strategies:

1. Random Agent
2. Greedy Agent
3. Q-Learning Agent

The first two agents provide baseline strategies for comparison. Q-Learning is the first reinforcement learning algorithm used by the project.

---

# 1. Random Agent

The Random Agent selects an action randomly from the available action space.

The V1 action space contains:

```text
UP
DOWN
LEFT
RIGHT
```

The agent does not consider:

* Distance to dirty cells
* Obstacles
* Battery
* Previous actions
* Rewards
* Learned experience

Conceptually:

```text
State
  ↓
Random action
  ↓
Environment
```

### Purpose

The Random Agent provides a simple baseline.

A learning algorithm should perform substantially better than a random strategy if learning is successful.

### Current Result

The current experiment produced:

```text
Episodes:              100
Average reward:        -30.00
Average steps:          25.00
Average cleaned cells:   0.00
Completion rate:        0.00%
Average invalid moves:  5.00
```

The Random Agent therefore provides a weak baseline, as expected.

---

# 2. Greedy Agent

The Greedy Agent chooses actions according to the nearest dirty cell.

It uses Manhattan distance:

```text
distance =
abs(robot_row - dirty_row)
+
abs(robot_column - dirty_column)
```

The agent attempts to move toward the closest remaining dirty cell.

Conceptually:

```text
Current state
     ↓
Find nearest dirty cell
     ↓
Choose direction
     ↓
Environment
```

### Advantages

The Greedy Agent:

* Does not require training
* Uses the current environment state
* Usually finds short paths
* Provides a strong deterministic baseline

### Limitations

The Greedy Agent does not learn.

It also does not explicitly optimize long-term reward.

Its behavior is based only on the current dirty-cell configuration.

---

# 3. Q-Learning

Q-Learning is the primary reinforcement learning algorithm in V1.

The agent maintains a Q-table:

```text
Q(state, action)
```

The Q-value estimates the expected future reward of performing an action from a given state.

The standard update rule is:

```text
Q(s,a) ← Q(s,a)
         + α [r + γ max Q(s',a') - Q(s,a)]
```

where:

```text
α = learning rate
γ = discount factor
r = reward
s = current state
a = selected action
s' = next state
```

---

## Exploration

V1 uses epsilon-greedy action selection.

With probability `epsilon`:

```text
Choose a random action
```

Otherwise:

```text
Choose the action with the highest Q-value
```

During training, epsilon decreases over time.

The current training experiment ends with:

```text
Final epsilon: 0.0100
```

This means the trained agent primarily exploits the learned Q-table.

---

## Training

The current V1 Q-Learning experiment uses:

```text
Training episodes: 1000
Final epsilon:     0.0100
Q-table entries:   10816
```

The agent interacts with the environment repeatedly and updates its Q-values after each transition.

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
Next step
```

---

## Evaluation

After training, the Q-Learning agent is evaluated separately.

During evaluation, exploration is disabled:

```text
epsilon = 0
```

This means the agent selects the learned best action instead of randomly exploring.

Current evaluation:

```text
Episodes:              100
Average reward:         40.00
Average steps:          10.00
Average cleaned cells:   3.00
Completion rate:       100.00%
Average invalid moves:   0.00
```

---

# Algorithm Comparison

| Agent      | Learning | Uses Distance | Uses Experience | Current Completion |
| ---------- | -------- | ------------- | --------------- | -----------------: |
| Random     | No       | No            | No              |                 0% |
| Greedy     | No       | Yes           | No              |               100% |
| Q-Learning | Yes      | Indirectly    | Yes             |               100% |

The current V1 experiments demonstrate that Q-Learning successfully learns an effective policy for the configured environment.

---

# V1 Algorithm Architecture

```text
                    Cleaning Environment
                           │
             ┌─────────────┼─────────────┐
             │             │             │
          Random         Greedy      Q-Learning
             │             │             │
             │             │         Q-table
             │             │             │
             └─────────────┼─────────────┘
                           │
                       Action
                           │
                           ↓
                    Environment
                           │
                  State + Reward
```

The three agents share the same environment interface, making their performance directly comparable.
