# Experiments

## Overview

Version 1 uses controlled experiments to compare different agent strategies in the autonomous cleaning environment.

The main objective is to determine whether the reinforcement learning agent can learn an effective cleaning policy and outperform a random baseline.

The current experiments compare:

* Random Agent
* Greedy Agent
* Q-Learning Agent

---

# 1. Random Agent Baseline

Command:

```bash
python src/cleaning_vehicle/experiments/random_baseline.py
```

Current result:

```text
Random Agent Baseline
=====================
Episodes:              100
Average reward:        -30.00
Average steps:          25.00
Average cleaned cells:   0.00
Completion rate:         0.00%
Average invalid moves:   5.00
```

### Interpretation

The Random Agent performs poorly.

It receives a negative reward because random movement causes:

* Unnecessary movement
* Invalid actions
* Failure to reach dirty cells efficiently

The agent provides the lower baseline for comparison.

---

# 2. Greedy Agent Baseline

Command:

```bash
python src/cleaning_vehicle/experiments/greedy_baseline.py
```

Current result:

```text
Greedy Agent Baseline
=====================
Episodes:              100
Average reward:         35.00
Average steps:          15.00
Average cleaned cells:   3.00
Completion rate:       100.00%
Average invalid moves:   0.00
```

### Interpretation

The Greedy Agent performs very well because it directly uses the locations of dirty cells.

It:

* Cleans all dirty cells
* Completes every episode
* Makes no invalid movements
* Requires approximately 15 steps

This provides a strong non-learning baseline.

---

# 3. Q-Learning Training

Command:

```bash
python src/cleaning_vehicle/experiments/q_learning_baseline.py
```

Training configuration:

```text
Episodes:       1000
Final epsilon:  0.0100
Q-table size:   10816
```

The Q-table contains 10,816 learned state entries after training.

---

# 4. Q-Learning Evaluation

The trained Q-Learning policy is evaluated separately with exploration disabled.

Current result:

```text
Q-Learning Evaluation
=====================
Episodes:              100
Average reward:          40.00
Average steps:           10.00
Average cleaned cells:    3.00
Completion rate:        100.00%
Average invalid moves:    0.00
```

---

# 5. Comparison

| Metric                | Random | Greedy | Q-Learning |
| --------------------- | -----: | -----: | ---------: |
| Episodes              |    100 |    100 |        100 |
| Average reward        | -30.00 |  35.00 |  **40.00** |
| Average steps         |  25.00 |  15.00 |  **10.00** |
| Average cleaned cells |   0.00 |   3.00 |   **3.00** |
| Completion rate       |     0% |   100% |   **100%** |
| Invalid moves         |   5.00 |   0.00 |   **0.00** |

---

# 6. Results Analysis

The experiments demonstrate three distinct behaviors.

### Random Agent

```text
Poor performance
↓
0% completion
↓
Many invalid actions
```

This establishes a weak baseline.

### Greedy Agent

```text
Uses current dirt locations
↓
Short direct paths
↓
100% completion
```

The Greedy Agent is a strong heuristic baseline.

### Q-Learning Agent

```text
Experience
↓
Q-table updates
↓
Learned policy
↓
100% completion
↓
10 average steps
```

The trained Q-Learning agent achieves a higher average reward and fewer steps than the Greedy Agent in the current experiment.

---

# 7. Reward Interpretation

The Q-Learning evaluation result is:

```text
Average reward = 40.00
```

For a typical successful episode:

```text
10 movements × -1 = -10
3 cleaned cells × +10 = +30
completion bonus    = +20

Total               = +40
```

This matches the measured result.

---

# 8. Reproducibility

The experiments should be run from the project root:

```bash
cd Autonomous-Cleaning-Vehicle
```

Then:

```bash
python src/cleaning_vehicle/experiments/random_baseline.py
python src/cleaning_vehicle/experiments/greedy_baseline.py
python src/cleaning_vehicle/experiments/q_learning_baseline.py
```

The experiment scripts are intended to provide simple, repeatable V1 performance measurements.

---

# 9. Testing Status

The complete V1 test suite currently passes:

```text
214 passed
```

Command:

```bash
pytest -q
```

This confirms that the current implementation and its unit tests are consistent with the V1 behavior.

---

# 10. V1 Experimental Conclusion

The current results demonstrate that the V1 environment is suitable for reinforcement learning experimentation.

The Q-Learning agent successfully learned a policy that:

* Cleans all three dirty cells
* Completes 100% of evaluation episodes
* Uses no invalid movements
* Requires approximately 10 steps
* Achieves an average reward of 40.00

The next versions can therefore focus on increasing environmental complexity rather than replacing the V1 foundation.
