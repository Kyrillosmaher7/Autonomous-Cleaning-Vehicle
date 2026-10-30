# Action Space

## 1. Overview

The action space defines the actions available to the autonomous cleaning vehicle.

Version 1 uses a discrete four-action movement space:

```text
A = {UP, DOWN, LEFT, RIGHT}
```

The vehicle can move one grid cell at a time.

---

## 2. Action Enum

Actions are represented by the `Action` enum.

```python
from cleaning_vehicle.environment.actions import Action
```

The available actions are:

```python
Action.UP
Action.DOWN
Action.LEFT
Action.RIGHT
```

Each action contains a position delta:

```text
Action      Delta
---------------------
UP          (-1,  0)
DOWN        ( 1,  0)
LEFT        ( 0, -1)
RIGHT       ( 0,  1)
```

The delta is available through:

```python
action.delta
```

---

## 3. Grid Coordinate System

The project uses:

```text
(row, column)
```

coordinates.

Rows increase downward.

Columns increase to the right.

Example:

```text
             column
          0   1   2   3   4
        ┌───┬───┬───┬───┬───┐
row 0   │   │   │   │   │   │
        ├───┼───┼───┼───┼───┤
row 1   │   │   │   │   │   │
        ├───┼───┼───┼───┼───┤
row 2   │   │   │ R │   │   │
        ├───┼───┼───┼───┼───┤
row 3   │   │   │   │   │   │
        └───┴───┴───┴───┴───┘
```

If the vehicle is at:

```text
(2, 2)
```

then:

```text
UP    → (1, 2)
DOWN  → (3, 2)
LEFT  → (2, 1)
RIGHT → (2, 3)
```

---

## 4. UP

The `UP` action moves the vehicle one row upward.

```python
Action.UP.delta
```

returns:

```text
(-1, 0)
```

Example:

```text
Current:
(2, 2)

UP:

(1, 2)
```

---

## 5. DOWN

The `DOWN` action moves the vehicle one row downward.

```python
Action.DOWN.delta
```

returns:

```text
(1, 0)
```

Example:

```text
Current:
(2, 2)

DOWN:

(3, 2)
```

---

## 6. LEFT

The `LEFT` action moves the vehicle one column to the left.

```python
Action.LEFT.delta
```

returns:

```text
(0, -1)
```

Example:

```text
Current:
(2, 2)

LEFT:

(2, 1)
```

---

## 7. RIGHT

The `RIGHT` action moves the vehicle one column to the right.

```python
Action.RIGHT.delta
```

returns:

```text
(0, 1)
```

Example:

```text
Current:
(2, 2)

RIGHT:

(2, 3)
```

---

## 8. Action Space Summary

The complete V1 action space is:

```text
┌────────────┬───────────────┐
│ Action     │ Position Delta│
├────────────┼───────────────┤
│ UP         │ (-1,  0)      │
│ DOWN       │ ( 1,  0)      │
│ LEFT       │ ( 0, -1)      │
│ RIGHT      │ ( 0,  1)      │
└────────────┴───────────────┘
```

Therefore:

```text
|A| = 4
```

---

## 9. Action Selection

The agent is responsible for selecting an action.

The interface is:

```python
action = agent.select_action(state)
```

The selected value must be an `Action`.

For example:

```python
action = Action.RIGHT
```

The agent does not directly modify the state.

Instead, it passes the action to the environment.

```text
State
  │
  ▼
Agent
  │
  │ select_action()
  ▼
Action
  │
  ▼
Environment
```

---

## 10. Action Execution

The environment converts the action into a candidate next position.

The calculation is:

```text
next_row = row + row_delta
next_col = column + column_delta
```

For example:

```text
Current position = (5, 5)
Action           = RIGHT
Delta            = (0, 1)

Next position:
(5 + 0, 5 + 1)
= (5, 6)
```

---

## 11. Boundary Handling

The environment prevents movement outside the grid.

For a `10 × 10` grid, valid coordinates are:

```text
row    = 0 ... 9
column = 0 ... 9
```

Suppose the vehicle is at:

```text
(0, 0)
```

and selects:

```text
UP
```

The candidate position is:

```text
(-1, 0)
```

which is outside the grid.

The action is therefore invalid.

The vehicle remains at:

```text
(0, 0)
```

---

## 12. Obstacle Handling

The environment also prevents movement onto static obstacles.

Example:

```text
Current position:

. . . . .
. . R # .
. . . . .
```

If the vehicle selects:

```text
RIGHT
```

the candidate position is the obstacle cell.

The movement is rejected.

The vehicle remains in its current position.

The agent therefore cannot bypass the environment's physical constraints.

---

## 13. Valid and Invalid Actions

An action is valid when its destination:

1. Is inside the grid.
2. Is not an obstacle.

Conceptually:

```text
Action
  │
  ▼
Calculate next position
  │
  ▼
Inside grid?
  │
  ├── No ───────► Invalid
  │
  ▼
Obstacle?
  │
  ├── Yes ──────► Invalid
  │
  ▼
Valid
```

---

## 14. Invalid Action Behavior

An invalid action does not change the vehicle position.

For example:

```text
Before:

position = (0, 0)

Action:

UP

After:

position = (0, 0)
```

The environment also applies the configured invalid-action reward.

The V1 default is:

```text
Invalid movement → -2
```

This discourages agents from repeatedly selecting impossible movements.

---

## 15. Valid Movement Reward

A valid movement has the default movement cost:

```text
Valid movement → -1
```

This encourages agents to reach dirty cells efficiently rather than wandering unnecessarily.

For example:

```text
State
  ↓
RIGHT
  ↓
Valid movement
  ↓
Reward = -1
```

If the destination is also dirty, the cleaning reward is added.

---

## 16. Cleaning Through Actions

There is no separate `CLEAN` action in V1.

Cleaning occurs automatically when the vehicle moves onto a dirty cell.

Example:

```text
Current:

position = (2, 2)

dirty_cells = {(2, 3)}

Action:

RIGHT
```

The vehicle moves to:

```text
(2, 3)
```

The dirty cell is then removed.

Therefore:

```text
Movement + Dirty destination
            ↓
          Clean
```

This keeps the V1 action space small.

---

## 17. Completion

If the final dirty cell is cleaned, the environment marks the episode as complete.

The transition becomes:

```text
State
  ↓
Action
  ↓
Move onto final dirty cell
  ↓
Clean cell
  ↓
No dirty cells remain
  ↓
done = True
```

The default completion bonus is:

```text
+20
```

Therefore a final cleaning action can receive:

```text
movement reward
+
cleaning reward
+
completion reward
```

With the V1 defaults:

```text
-1 + 10 + 20 = +29
```

---

## 18. Battery Interaction

Battery is part of the V1 state:

```text
State = (position, dirty_cells, battery)
```

Actions therefore operate within an energy-aware environment.

A movement can consume battery according to the battery component's configured policy.

Conceptually:

```text
State
  │
  ├── Position
  ├── Dirt
  └── Battery
       │
       ▼
     Action
       │
       ▼
  Environment
       │
       ├── Validate movement
       ├── Consume energy
       └── Produce next state
```

The action itself does not contain battery logic.

Battery rules belong to the environment/battery layer.

---

## 19. Charging Station

V1 includes a charging-station concept.

The charging station is not represented as an additional movement action.

The vehicle still uses:

```text
UP
DOWN
LEFT
RIGHT
```

to navigate to the charging station.

Charging is therefore an environmental effect associated with the vehicle being at the charging location rather than a fifth movement direction.

Conceptually:

```text
Move
  ↓
Charging station
  ↓
Battery recharge
```

This preserves the simple four-action V1 action space.

---

## 20. No Diagonal Movement

V1 does not support diagonal actions.

The following actions do not exist:

```text
UP_LEFT
UP_RIGHT
DOWN_LEFT
DOWN_RIGHT
```

Movement is restricted to the four cardinal directions.

This creates a standard 4-connected grid.

---

## 21. No Stay Action

V1 does not explicitly define a `STAY` or `WAIT` action.

The vehicle moves only through:

```text
UP
DOWN
LEFT
RIGHT
```

An invalid action may result in the vehicle remaining in the same position, but this is not considered an intentional `STAY` action.

---

## 22. Q-Learning Action Values

For tabular Q-Learning, each state has four possible action values:

```text
Q(S, UP)
Q(S, DOWN)
Q(S, LEFT)
Q(S, RIGHT)
```

The agent can therefore represent its action preferences as:

```text
              State S
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
      UP       DOWN       LEFT       RIGHT
       │         │          │          │
       ▼         ▼          ▼          ▼
      Q₁        Q₂         Q₃         Q₄
```

The greedy policy selects the action with the highest Q-value.

---

## 23. Epsilon-Greedy Selection

During Q-Learning training, the agent uses epsilon-greedy exploration.

With probability:

```text
ε
```

the agent explores by selecting an action.

Otherwise it selects the action with the highest current Q-value.

Conceptually:

```text
                 State
                   │
                   ▼
              Random choice?
              /           \
            Yes             No
             │               │
             ▼               ▼
        Explore          Best Q-value
             │               │
             └───────┬───────┘
                     ▼
                   Action
```

During final evaluation, epsilon is set to:

```text
0.0
```

so the learned policy acts greedily.

---

## 24. Action Validation

The environment validates that the supplied action is actually an `Action`.

Invalid types raise:

```python
TypeError
```

For example, an agent should not return:

```python
"RIGHT"
```

or:

```python
(0, 1)
```

Instead it must return:

```python
Action.RIGHT
```

This provides a clear contract between agents and the environment.

---

## 25. Action Contract

The V1 agent/environment contract is:

```text
Agent
  │
  │ select_action(state)
  ▼
Action
  │
  │ environment.step(state, action)
  ▼
Next State
Reward
Done
```

The environment owns the consequences of the action.

The agent owns the decision.

---

## 26. V1 Action Space

The final V1 action space is intentionally minimal:

```text
A = {
    UP,
    DOWN,
    LEFT,
    RIGHT
}
```

The four actions are sufficient to support:

* Grid navigation
* Obstacle avoidance
* Boundary handling
* Dirt collection
* Charging-station navigation
* Battery-aware movement
* Random policies
* Greedy policies
* Q-Learning
* SARSA

The action space can later be extended if the environment requires additional behaviors.

---

## 27. Future Action Extensions

Future versions may introduce actions or action semantics such as:

```text
CHARGE
WAIT
CLEAN
```

or more advanced movement controls.

However, these are intentionally excluded from the V1 movement action space.

Keeping V1 at four discrete movement actions makes the problem easier to understand, test, and solve with tabular reinforcement learning.

---

## 28. Final Definition

The V1 action space is:

```text
┌─────────────────────────────┐
│       V1 Action Space       │
├─────────────────────────────┤
│ UP       → (-1,  0)         │
│ DOWN     → ( 1,  0)         │
│ LEFT     → ( 0, -1)         │
│ RIGHT    → ( 0,  1)         │
└─────────────────────────────┘
```

Mathematically:

```text
A = {↑, ↓, ←, →}
```

The agent chooses one action from this set at every decision step, and the environment determines whether that action is physically valid and what state, reward, and termination result from it.
