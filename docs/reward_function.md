# Reward Function

## Overview

The reward function defines how the autonomous cleaning vehicle evaluates the quality of its actions.

For Version 1, the reward design is intentionally simple and focuses on four events:

* Valid movement
* Invalid movement
* Cleaning a dirty cell
* Completing the cleaning task

The reward calculation is implemented through `RewardCalculator` and `RewardConfig`, allowing the reward policy to be changed without modifying the environment logic.

---

## Reward Configuration

The current V1 configuration is:

| Event            | Reward |
| ---------------- | -----: |
| Valid movement   |   `-1` |
| Invalid movement |   `-2` |
| Clean dirty cell |  `+10` |
| Complete task    |  `+20` |

Therefore:

```text
clean       = +10
move        = -1
invalid     = -2
completion  = +20
```

---

## 1. Valid Movement

When the vehicle successfully moves to a valid neighboring cell:

```text
Reward = -1
```

The movement penalty encourages the agent to reach dirty cells using short paths rather than moving unnecessarily.

For example:

```text
State:  (2,2)
Action: RIGHT
Next:   (2,3)

Reward = -1
```

---

## 2. Invalid Movement

An action is invalid when the requested position is:

* Outside the grid boundary
* Occupied by a static obstacle

The vehicle remains in its current position.

```text
Reward = -2
```

The larger penalty discourages repeatedly attempting impossible movements.

---

## 3. Cleaning a Dirty Cell

When the vehicle moves onto a dirty cell, the cell is removed from the remaining dirty-cell set.

The agent receives:

```text
Movement reward = -1
Cleaning reward = +10
```

Therefore, a normal movement onto a dirty cell produces:

```text
-1 + 10 = +9
```

---

## 4. Completing the Task

When the last dirty cell is cleaned, the environment marks the episode as complete and provides an additional:

```text
Completion reward = +20
```

Therefore, the final cleaning transition produces:

```text
Movement       -1
Cleaning      +10
Completion    +20
------------------
Total         +29
```

---

## 5. Example Complete Episode

Suppose the vehicle cleans three dirty cells using ten valid movements.

The approximate reward can be calculated as:

```text
Movement:
10 × -1 = -10

Cleaning:
3 × +10 = +30

Completion:
1 × +20 = +20

Total:
-10 + 30 + 20 = +40
```

This matches the current Q-Learning evaluation result:

```text
Average reward: 40.00
Average steps:  10.00
Cleaned cells: 3.00
```

---

## 6. Reward Design Philosophy

The reward function balances three objectives:

### Efficiency

Every movement has a small negative cost.

This discourages unnecessarily long paths.

### Correct behavior

Invalid movements receive a stronger penalty.

This discourages attempts to move through obstacles or outside the grid.

### Task completion

Cleaning provides a strong positive reward, while completing the entire task provides an additional bonus.

This gives the learning agent a clear objective:

```text
Clean all dirt
      ↓
Use valid movements
      ↓
Avoid unnecessary movement
      ↓
Finish the episode
```

---

## 7. Configurability

The reward values are defined through `RewardConfig`.

Conceptually:

```python
RewardConfig(
    clean=10.0,
    move=-1.0,
    invalid=-2.0,
    completion=20.0,
)
```

This allows future experiments to modify the reward policy without changing the environment implementation.

For example, future versions may experiment with:

* Battery penalties
* Charging rewards
* Distance-based penalties
* Time penalties
* Energy consumption
* Dynamic obstacle penalties

---

## 8. V1 Reward Model

The V1 reward model can be summarized as:

```text
                 ┌── Invalid action ──→ -2
                 │
Action ──────────┼── Valid movement ──→ -1
                 │
                 └── Dirty cell ─────→ +10

Last dirty cell cleaned
          │
          └──────────────→ +20 completion bonus
```

The reward system is deliberately simple for V1 so that the effect of the learning algorithm can be evaluated clearly.
