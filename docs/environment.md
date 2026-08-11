# Environment

## 1. Overview

The environment represents the world in which the autonomous cleaning vehicle operates.

The environment is responsible for enforcing the rules of the simulation.

Its main responsibilities are:

* Grid boundaries
* Static obstacles
* Vehicle movement
* Dirty cells
* Cleaning
* Battery management
* Charging
* Reward calculation
* Episode termination

The agent does not directly modify the environment.

Instead, the agent selects an action and the environment processes it.

---

## 2. Environment- Agent Interaction

The fundamental interaction is:

```text id="5q6c8x"
        State
          │
          ▼
       Agent
          │
      Action
          │
          ▼
   ┌──────────────┐
   │ Environment  │
   └──────┬───────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 Next   Reward  Done
 State
```

This follows the standard reinforcement-learning interaction:

```text id="8r6y9v"
(S_t, A_t)
    ↓
Environment
    ↓
(S_t+1, R_t+1, Done)
```

---

# 3. Grid

The physical world is represented by the `Grid` component.

The grid contains:

```text id="7c4m5f"
Grid
├── Width
├── Height
├── Obstacles
└── Dirty Cells
```

The grid uses `(row, column)` coordinates.

For example:

```text id="wq2y5v"
5 × 5 grid

. . . . .
. # . D .
. . R . .
. . . # .
. . . . .
```

Where:

```text
. = Empty
# = Obstacle
D = Dirty
R = Vehicle
```

---

# 4. Grid Dimensions

The environment requires positive dimensions.

For example:

```python id="h5j4mc"
CleaningEnvironment(
    rows=10,
    cols=10,
)
```

represents:

```text
10 rows × 10 columns
```

Therefore the grid contains:

```text id="6t3y0x"
10 × 10 = 100 cells
```

For a `20 × 20` environment:

```text
20 × 20 = 400 cells
```

---

# 5. Position Validation

A position is valid when it is inside the grid.

For a `10 × 10` grid:

```text id="c8x1xk"
0 ≤ row < 10
0 ≤ column < 10
```

Therefore:

```text id="0l3b7f"
(0, 0)     → valid
(9, 9)     → valid
(10, 9)    → invalid
(9, 10)    → invalid
(-1, 0)    → invalid
```

The environment validates positions before allowing movement.

---

# 6. Obstacles

Obstacles are static blocked cells.

Example:

```text id="9j5d6w"
. . . . .
. # # . .
. . R . .
. . . . .
. . . . D
```

The vehicle cannot move onto:

```text
(1, 1)
(1, 2)
```

because they contain obstacles.

Obstacles are managed by the environment and are not selected as actions.

---

# 7. Traversability

A cell is traversable when:

```text id="6k5r8s"
Inside grid
     AND
Not an obstacle
```

Conceptually:

```python id="g6g5bh"
is_traversable(position)
```

returns `True` only when both conditions are satisfied.

This rule is used during movement.

---

# 8. Vehicle Movement

Movement is controlled through the `Action` enum.

The available actions are:

```text id="7n3f5c"
UP
DOWN
LEFT
RIGHT
```

Each action provides a position delta.

```text id="m7q9a4"
UP       → (-1,  0)
DOWN     → ( 1,  0)
LEFT     → ( 0, -1)
RIGHT    → ( 0,  1)
```

The environment calculates:

```text id="0g5e2s"
next_position =
    current_position + action.delta
```

---

# 9. Valid Movement

Suppose:

```text id="1r7v0y"
Current position = (2, 2)
Action = RIGHT
```

Then:

```text id="3q8m1w"
RIGHT = (0, 1)

next_position = (2, 3)
```

If `(2, 3)` is inside the grid and is not an obstacle, movement succeeds.

The resulting state contains:

```text id="z9w4y1"
position = (2, 3)
```

---

# 10. Invalid Movement

Movement is rejected when the destination is invalid.

Two major cases exist.

### Boundary

```text id="9c4k2p"
Current = (0, 0)
Action  = UP

Candidate = (-1, 0)
```

The candidate is outside the grid.

### Obstacle

```text id="0w5n8k"
Current = (2, 2)
Action  = RIGHT

Candidate = (2, 3)
```

If `(2, 3)` is an obstacle, the movement is rejected.

In both cases:

```text id="5y1n7r"
position remains unchanged
```

---

# 11. Cleaning

Dirty cells represent cells that must be cleaned.

When the vehicle reaches a dirty cell, the cell is automatically cleaned.

There is no separate `CLEAN` action in V1.

Example:

```text id="1c7x5m"
Before:

R D
```

The vehicle executes:

```text
RIGHT
```

and reaches the dirty cell.

After the transition:

```text id="4n6y2q"
R .
```

The dirty cell is removed from the state.

---

# 12. Cleaning Transition

Suppose:

```python id="h1g7y0"
state.position = (2, 2)

state.dirty_cells = frozenset({
    (2, 3),
})
```

The agent selects:

```python id="3k5p8m"
Action.RIGHT
```

The environment produces:

```python id="z7q3y2"
next_state.position == (2, 3)

next_state.dirty_cells == frozenset()
```

The cell has been cleaned.

---

# 13. Battery

The V1 state includes the current battery level.

Conceptually:

```text id="r4m8j6"
State
├── Position
├── Dirty Cells
└── Battery
```

The battery represents remaining available energy.

Example:

```text id="w3k9p1"
Battery = 20
```

means the vehicle currently has 20 energy units available.

---

# 14. Battery Consumption

Movement can consume battery according to the battery component's configured policy.

Conceptually:

```text id="k4j8n2"
Before action:

Battery = B

       ↓
     Action

       ↓

After action:

Battery = B - cost
```

Battery behavior is kept separate from the action definition.

For example:

```text id="6h3x9v"
RIGHT
```

describes movement.

The battery system determines the energy consequences.

---

# 15. Battery Capacity

The battery has a maximum capacity.

Conceptually:

```text id="0k8r4m"
0 ≤ battery ≤ maximum_capacity
```

The battery component is responsible for enforcing this constraint.

Charging cannot increase the battery above its configured maximum.

---

# 16. Charging Station

The environment can contain a charging station.

The charging station represents a position where the vehicle can restore battery energy.

The vehicle reaches the station through normal movement:

```text id="c5p2n8"
UP
DOWN
LEFT
RIGHT
```

There is no separate movement action called `CHARGE`.

Conceptually:

```text id="q9m4v6"
Vehicle
   │
   │ move
   ▼
Charging Station
   │
   ▼
Recharge Battery
```

---

# 17. Charging Behavior

When the vehicle is at the charging station, the charging component can restore the battery according to the configured policy.

The important separation is:

```text id="x2n7m4"
Action
   ↓
Movement

Charging Station
   ↓
Battery behavior
```

This prevents charging logic from being mixed into the `Action` enum.

---

# 18. Reward Calculation

The environment delegates reward calculation to `RewardCalculator`.

The default V1 configuration is:

```text id="1v5q8n"
Valid movement       -1
Invalid movement     -2
Clean dirty cell    +10
Completion           +20
```

These values are configurable through `RewardConfig`.

---

# 19. Valid Movement Reward

Every valid movement receives:

```text id="p3k8x2"
-1
```

This creates a small cost for movement.

The purpose is to encourage the agent to complete the cleaning task efficiently.

A policy that takes unnecessary steps receives lower cumulative reward.

---

# 20. Invalid Movement Reward

An invalid movement receives:

```text id="j7q2v5"
-2
```

The vehicle remains in the same position.

This discourages the agent from repeatedly attempting impossible movements.

---

# 21. Cleaning Reward

When a dirty cell is successfully cleaned:

```text id="x8m2q4"
+10
```

is added to the reward.

Therefore a valid movement onto a dirty cell normally produces:

```text id="h4p7y1"
Movement
-1

Cleaning
+10

Total
+9
```

before any completion bonus.

---

# 22. Completion Reward

When the final dirty cell is cleaned:

```text id="v9q3m6"
+20
```

completion reward is added.

For the final cleaning transition:

```text id="k2x8p4"
Movement       -1
Cleaning      +10
Completion    +20
----------------
Total         +29
```

This gives the agent a strong incentive to finish the task.

---

# 23. Environment Step

The primary RL interface is:

```python id="u5m9r2"
environment.step(state, action)
```

It returns:

```text id="7x4n8p"
next_state
reward
done
```

The complete transition is:

```text id="j8q2m5"
State + Action
      │
      ▼
┌──────────────────────┐
│      Environment     │
├──────────────────────┤
│ Validate action      │
│ Calculate position   │
│ Check boundary       │
│ Check obstacle       │
│ Move vehicle         │
│ Clean destination    │
│ Update battery       │
│ Apply charging       │
│ Calculate reward     │
│ Check completion     │
└──────────┬───────────┘
           │
           ▼
Next State + Reward + Done
```

---

# 24. Complete Transition Example

Consider:

```text id="n3c7v8"
Position:
(2, 2)

Dirty:
{(2, 3)}

Battery:
20
```

The agent selects:

```text
RIGHT
```

### Step 1 — Calculate destination

```text id="6v2x9m"
(2, 2) + (0, 1)
= (2, 3)
```

### Step 2 — Validate

```text id="r5k1q8"
Inside grid?      Yes
Obstacle?         No
```

### Step 3 — Move

```text id="w7p4n2"
Position = (2, 3)
```

### Step 4 — Clean

```text id="c8m5y1"
Dirty cells = {}
```

### Step 5 — Reward

```text id="z2q6v4"
Movement      -1
Cleaning     +10
Completion   +20
-----------------
Total         29
```

### Step 6 — Termination

```text id="p4x7m8"
done = True
```

---

# 25. Episode Termination

The main V1 completion condition is:

```text id="n8q3v6"
No dirty cells remain.
```

Therefore:

```python id="c6m2x8"
environment.is_done(state)
```

returns `True` when:

```python id="d9p4k1"
len(state.dirty_cells) == 0
```

The episode runner then stops the episode.

---

# 26. Environment Responsibilities

The environment owns the consequences of actions.

### Environment responsibilities

```text id="8k4m2p"
✓ Validate positions
✓ Validate obstacles
✓ Apply movement
✓ Prevent invalid movement
✓ Clean dirty cells
✓ Manage battery interaction
✓ Handle charging
✓ Calculate rewards
✓ Detect completion
```

### Agent responsibilities

```text id="m7x3q9"
✓ Observe state
✓ Select action
✓ Learn a policy when applicable
```

This separation is fundamental to the project.

---

# 27. Grid vs Environment

The `Grid` and `CleaningEnvironment` have different responsibilities.

### Grid

The grid represents the physical world:

```text id="r2k7v5"
Grid
├── Dimensions
├── Obstacles
├── Dirty cells
├── Cell validation
├── Traversability
└── Rendering
```

### CleaningEnvironment

The environment provides the RL interaction:

```text id="q6m3x8"
CleaningEnvironment
├── Movement
├── State transitions
├── Rewards
├── Cleaning
├── Battery interaction
├── Charging
└── Episode completion
```

The distinction prevents the grid representation from becoming responsible for reinforcement-learning behavior.

---

# 28. Environment Determinism in V1

The V1 environment is primarily deterministic.

Given the same:

```text id="w9p4k2"
State
+
Action
+
Environment configuration
```

the environment should produce the same transition.

For example:

```text id="n6x2m7"
State = S
Action = RIGHT

        ↓

Next State = S'
Reward = R
Done = D
```

This property is particularly useful for unit testing and debugging Q-Learning.

---

# 29. Environment Testing

The environment is covered by unit tests for important behaviors.

Tests include concepts such as:

```text id="p8q4m1"
✓ Valid movement
✓ Boundary prevention
✓ Obstacle prevention
✓ Invalid movement
✓ Dirty-cell cleaning
✓ Completion detection
✓ Reward calculation
✓ Battery behavior
✓ Charging behavior
```

The project currently has:

```text id="x7m3v9"
214 tests passing
```

across the complete V1 test suite.

---

# 30. Environment and Agents

The same environment can be used by multiple agents.

```text id="v4q8m2"
             Cleaning Environment
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
    Random        Greedy      Q-Learning
```

This is important because it allows algorithms to be compared under the same environment rules.

The environment therefore does not contain algorithm-specific logic.

---

# 31. Environment and Evaluation

The evaluation layer repeatedly runs the environment with an agent.

Conceptually:

```text id="m5x8q3"
Evaluation
    │
    ├── Episode 1
    ├── Episode 2
    ├── Episode 3
    ├── ...
    └── Episode N
```

Metrics are aggregated across episodes.

The current V1 evaluation metrics include:

```text
Average reward
Average steps
Average cleaned cells
Completion rate
Average invalid actions
```

---

# 32. Environment Design Principle

The most important rule is:

> **The environment is the authority on what is physically possible.**

An agent can request:

```text
UP
```

but the environment decides whether moving UP is valid.

An agent cannot bypass:

* Grid boundaries
* Obstacles
* Battery constraints
* Other environment rules

This ensures that learned behavior is evaluated against the actual simulation rules.

---

# 33. V1 Environment Summary

The final V1 environment can be summarized as:

```text id="q3m8v5"
┌──────────────────────────────────┐
│          V1 Environment          │
├──────────────────────────────────┤
│ Grid                             │
│   ├── Dimensions                 │
│   ├── Obstacles                  │
│   └── Dirty cells                │
│                                  │
│ Vehicle                          │
│   ├── Position                   │
│   └── Battery                    │
│                                  │
│ Charging                         │
│   └── Charging station           │
│                                  │
│ Dynamics                         │
│   ├── Movement                   │
│   ├── Cleaning                   │
│   ├── Battery interaction        │
│   └── Charging                   │
│                                  │
│ RL Interface                     │
│   ├── State                      │
│   ├── Action                     │
│   ├── Reward                     │
│   └── Done                       │
└──────────────────────────────────┘
```

The environment therefore provides the complete simulation boundary for the V1 autonomous cleaning vehicle.

---

# 34. V1 Transition Contract

The final environment contract is:

```text id="f7m2x9"
                State
                  │
                  ▼
                Action
                  │
                  ▼
        ┌──────────────────┐
        │ CleaningEnvironment│
        └────────┬─────────┘
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   Next State  Reward     Done
```

Formally:

```text id="n2q6v8"
Environment.step(S, A)
        →
(S', R, Done)
```

This transition is the central interface connecting the environment to every V1 reinforcement-learning agent.
