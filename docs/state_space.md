# State Space

## 1. Overview

The state represents the current situation of the autonomous cleaning vehicle.

In Version 1, the state contains the dynamic information required by the agent to make decisions:

```text
State
├── Position
├── Remaining Dirty Cells
└── Battery
```

The state is immutable and hashable so it can safely be used as a key in the tabular Q-Learning implementation.

---

## 2. State Representation

The V1 state is represented by:

```python
State(
    position=(row, column),
    dirty_cells=frozenset(...),
    battery=...
)
```

Conceptually:

```text
S = (P, D, B)
```

where:

* `P` = vehicle position
* `D` = remaining dirty cells
* `B` = current battery level

---

## 3. Position

The vehicle position is represented as:

```text
(row, column)
```

The coordinate system is:

```text
        columns →
       0   1   2   3

row 0  .   .   .   .
 ↓
row 1  .   .   .   .
row 2  .   R   .   .
row 3  .   .   .   .
```

Rows increase downward.

Columns increase to the right.

For example:

```python
position = (2, 3)
```

means:

```text
row    = 2
column = 3
```

---

## 4. Dirty Cells

The state stores the cells that still require cleaning.

Example:

```python
dirty_cells = frozenset({
    (1, 3),
    (4, 4),
    (0, 0),
})
```

This means three dirty cells remain.

The number of remaining dirty cells is:

```python
len(state.dirty_cells)
```

The state provides:

```python
state.remaining_dirty
```

for this purpose.

---

## 5. Why Dirty Cells Are Part of the State

The agent needs to know what cleaning work remains.

Consider:

```text
State A:
position = (2, 2)
dirty   = {(2, 5)}

State B:
position = (2, 2)
dirty   = {(0, 0)}
```

The vehicle is in the same position, but the optimal action can be different because the remaining task is different.

Therefore position alone is insufficient.

The dirty-cell configuration must be part of the RL state.

---

## 6. Battery

V1 also includes the current battery level:

```python
battery = 20
```

The battery represents the remaining available energy of the vehicle.

Therefore:

```text
S = (position, dirty_cells, battery)
```

Two states with identical positions and dirty cells but different battery levels are different RL states.

For example:

```text
State A:
position = (2, 2)
dirty    = {(4, 4)}
battery  = 20

State B:
position = (2, 2)
dirty    = {(4, 4)}
battery  = 5
```

These states must not be treated as identical because the vehicle has different remaining energy.

---

## 7. Battery and Decision Making

Battery introduces an additional constraint into decision making.

Without battery:

```text
Where should I move?
```

With battery:

```text
Where should I move,
and do I have enough energy
to continue operating?
```

This becomes particularly important when charging stations are introduced.

The future V1/V2 decision process can therefore consider:

```text
Current position
       +
Remaining dirt
       +
Battery level
       ↓
Choose action
```

---

## 8. Immutable State

`State` is implemented as an immutable dataclass.

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class State:
    ...
```

This means an existing state is not modified.

Instead, operations return a new state.

For example:

```python
next_state = state.move(new_position)
```

rather than modifying:

```python
state.position
```

directly.

---

## 9. Why Immutability Matters

Immutability is particularly useful for reinforcement learning.

Q-Learning requires states to be stored and compared repeatedly.

A state can therefore safely be used as:

```python
Q[state]
```

without worrying that the state will later change and invalidate the dictionary key.

---

## 10. State Hashing

The state hash is based on the complete state representation:

```text
(position, dirty_cells, battery)
```

Conceptually:

```python
hash(
    (
        state.position,
        state.dirty_cells,
        state.battery,
    )
)
```

Therefore:

```text
Same position
+
Same dirty cells
+
Same battery
=
Same state
```

while:

```text
Same position
+
Same dirty cells
+
Different battery
=
Different state
```

---

## 11. State Tuple Representation

The state provides an explicit tuple representation through:

```python
state.as_tuple()
```

The V1 representation is:

```python
(
    state.position,
    state.dirty_cells,
    state.battery,
)
```

This representation is useful for:

* Q-table operations
* Debugging
* Testing
* Serialization in future versions
* Inspecting learned states

---

## 12. State Transitions

The state changes through environment transitions.

The general RL transition is:

```text
S_t + A_t
      │
      ▼
Environment
      │
      ▼
S_(t+1) + R_(t+1)
```

For V1:

```text
Current State
     │
     ├── position
     ├── dirty cells
     └── battery
     │
     ▼
   Action
     │
     ▼
Environment
     │
     ├── validate movement
     ├── update position
     ├── clean destination
     ├── update battery
     ├── calculate reward
     └── check completion
     │
     ▼
Next State
```

---

## 13. Movement Transition

Suppose:

```text
Current position = (2, 2)
Action           = RIGHT
```

The action delta is:

```text
RIGHT = (0, 1)
```

Therefore:

```text
(2, 2) + (0, 1)
= (2, 3)
```

The next state contains the updated position.

If movement consumes energy, the battery is also updated.

---

## 14. Cleaning Transition

If the vehicle moves onto a dirty cell:

```text
Before:

position = (2, 2)
dirty    = {(2, 3)}
```

After moving RIGHT:

```text
position = (2, 3)
dirty    = {}
```

The dirty cell is removed from the state.

Therefore cleaning is represented directly through the state transition.

---

## 15. Invalid Movement

An invalid action does not produce an invalid position.

For example, if the vehicle is at:

```text
(0, 0)
```

and attempts:

```text
UP
```

the resulting position cannot become:

```text
(-1, 0)
```

Instead, the environment prevents the movement.

Conceptually:

```text
Invalid action
      ↓
Position unchanged
      ↓
Battery/reward handled by environment policy
```

This keeps the state valid.

---

## 16. Obstacles

Static obstacles are environment information rather than part of the V1 state.

For example:

```text
Grid:

. . . . .
. # . . .
. . R . .
. . . D .
. . . . .
```

The obstacle layout is maintained by the environment.

The state does not duplicate the obstacle set.

This avoids unnecessarily increasing the Q-table state representation.

---

## 17. State Space Size

The theoretical state space can grow quickly.

For a grid with:

```text
N = rows × columns
```

cells, the possible vehicle positions are approximately:

```text
N
```

and the possible dirty-cell configurations can be represented by subsets of the grid.

Ignoring obstacles and battery constraints for a simplified estimate:

```text
Position states ≈ N

Dirty configurations ≈ 2^N
```

With battery levels:

```text
State space ≈ N × 2^N × B
```

where `B` represents the number of possible battery levels.

This illustrates why tabular Q-Learning becomes increasingly expensive as the environment grows.

---

## 18. Example

For a small environment:

```text
Grid: 5 × 5

Position:
(2, 2)

Dirty cells:
{
    (0, 0),
    (4, 4)
}

Battery:
20
```

The state is:

```python
State(
    position=(2, 2),
    dirty_cells=frozenset({
        (0, 0),
        (4, 4),
    }),
    battery=20,
)
```

The agent observes this state and selects an action.

---

## 19. Terminal State

The primary V1 completion condition is:

```text
No dirty cells remain
```

Therefore:

```python
state.is_clean
```

is true when:

```python
len(state.dirty_cells) == 0
```

The environment then reports:

```text
done = True
```

The episode terminates.

---

## 20. State and Q-Learning

The Q-Learning agent estimates:

```text
Q(S, A)
```

where:

```text
S = (position, dirty_cells, battery)
A = {UP, DOWN, LEFT, RIGHT}
```

The learned table therefore maps state-action combinations to expected future rewards.

Conceptually:

```text
                 Actions
              ┌──────────────┐
State         │ U D L R      │
              └──────────────┘
                   │
                   ▼
              Q-values
```

Example:

```text
Q(state, UP)    =  4.2
Q(state, DOWN)  = -1.7
Q(state, LEFT)  =  2.1
Q(state, RIGHT) =  8.4
```

The agent can select `RIGHT` because it currently has the highest estimated value.

---

## 21. Design Principle

The V1 state follows a simple rule:

> **Store dynamic information required for decision making; keep static environment information outside the state.**

Therefore:

### Included

```text
✓ Position
✓ Remaining dirty cells
✓ Battery
```

### Kept in environment

```text
✓ Grid dimensions
✓ Static obstacles
✓ Environment configuration
```

This keeps the state representation explicit and manageable while providing the information required by the V1 agents.

---

## 22. Future Extensions

The state representation can be extended in later versions.

Possible V2 information includes:

```text
Dynamic obstacle information
Newly appearing dirt
Partial observations
Distance to charging station
Battery-related observations
```

Possible V3/V4 representations may additionally include:

```text
Sensor observations
Local grid observations
History / temporal information
Neural-network input tensors
```

The important architectural principle is to add information only when it is required by the agent's decision-making problem.

---

## 23. V1 Final State

The final V1 state is:

```text
┌─────────────────────────────┐
│            State            │
├─────────────────────────────┤
│ Position                    │
│ Remaining Dirty Cells       │
│ Battery                     │
└─────────────────────────────┘
```

Mathematically:

```text
S = (P, D, B)
```

This state representation is the foundation used by the V1 Random, Greedy, and Q-Learning agents.
