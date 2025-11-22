# Snake RL Environment - Interface Documentation

**Course:** COMPSCI 687 - Fall 2025  
**Authors:** [Add your group member names]

---

## Table of Contents
1. [Overview](#overview)
2. [MDP Formulation](#mdp-formulation)
3. [Environment Interface](#environment-interface)
4. [Usage Examples](#usage-examples)
5. [Implementation Notes](#implementation-notes)

---

## Overview

This Snake game environment is designed for training Reinforcement Learning algorithms. The environment implements a classic Snake game on an 8×8 grid with obstacles.

### Key Features
- **Grid Size:** 8×8
- **State Space:** 16,384 possible states
- **Action Space:** 4 actions (UP, DOWN, LEFT, RIGHT)
- **Obstacles:** 4 fixed obstacles at positions (2,2), (5,2), (2,5), (5,5)
- **Rendering:** Text mode or Pygame visualization

---

## MDP Formulation

### States (S)

The state is represented as a 5-tuple:

```
s = (head_x, head_y, food_x, food_y, direction)
```

Where:
- **head_x:** X-coordinate of snake head (0-7)
- **head_y:** Y-coordinate of snake head (0-7)
- **food_x:** X-coordinate of food (0-7)
- **food_y:** Y-coordinate of food (0-7)
- **direction:** Current direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)

**State Space Size:** 8 × 8 × 8 × 8 × 4 = **16,384 states**

### Actions (A)

Four discrete actions:

```
0 = UP      (move head up, y-1)
1 = DOWN    (move head down, y+1)
2 = LEFT    (move head left, x-1)
3 = RIGHT   (move head right, x+1)
```

**Action Space Size:** **4 actions**

### Rewards (R)

The reward function:

| Event | Reward | Description |
|-------|--------|-------------|
| Eat food | +10 | Snake head reaches food position |
| Step | -1 | Every movement (encourages efficiency) |
| Death | -20 | Hit wall, obstacle, or self |

**Death conditions:**
1. Head goes outside grid boundaries (0 ≤ x,y < 8)
2. Head collides with obstacle
3. Head collides with own body

### Transition Function (P)

The environment is **deterministic**. Given state s and action a:
- New head position is calculated based on action
- If death condition → episode terminates
- If food eaten → snake grows by 1, new food spawns
- Otherwise → snake moves (head advances, tail removed)

### Initial State Distribution

- Snake starts at center (4, 4) with length 3
- Random initial direction (0-3)
- Body extends opposite to direction
- Food spawns at random empty position

### Terminal States

Episode ends when:
1. Snake dies (wall, obstacle, or self-collision)
2. Maximum steps reached (200 steps)

### Discount Factor (γ)

Recommended: **γ = 0.99** (standard for episodic tasks)

---

## Environment Interface

### Creating the Environment

```python
from snake_env import SnakeEnv

# Text mode (fast, for training)
env = SnakeEnv(render_mode='text')

# Pygame mode (visual, for demonstration)
env = SnakeEnv(render_mode='pygame', cell_size=70)
```

### Core Methods

#### `reset() -> state`
Resets the environment to start a new episode.

```python
state = env.reset()
# Returns: (head_x, head_y, food_x, food_y, direction)
# Example: (4, 4, 6, 2, 3) means head at (4,4), food at (6,2), facing RIGHT
```

#### `step(action) -> (next_state, reward, done)`
Takes an action and returns the result.

```python
next_state, reward, done = env.step(action)

# next_state: new state tuple
# reward: -20, -1, or +10
# done: True if episode ended, False otherwise
```

#### `render(delay=0)`
Displays the current state.

```python
# Text mode
env.render()

# Pygame mode (with delay in milliseconds)
env.render(delay=200)
```

#### `close()`
Closes the environment and cleans up resources.

```python
env.close()  # Important for pygame mode!
```

### Helper Methods

#### `get_action_space_size() -> int`
Returns 4 (number of actions).

#### `get_state_space_size() -> int`
Returns 16,384 (total possible states).

#### `get_grid_size() -> int`
Returns 8 (grid dimension).

#### `get_obstacles() -> List[Tuple[int, int]]`
Returns list of obstacle positions: `[(2,2), (5,2), (2,5), (5,5)]`

#### `get_state_bounds() -> dict`
Returns min/max values for each state component.

```python
bounds = env.get_state_bounds()
# Returns:
# {
#   'head_x': (0, 7),
#   'head_y': (0, 7),
#   'food_x': (0, 7),
#   'food_y': (0, 7),
#   'direction': (0, 3)
# }
```

#### `get_episode_info() -> dict`
Returns current episode statistics.

```python
info = env.get_episode_info()
# Returns:
# {
#   'steps': 42,
#   'total_reward': -12,
#   'snake_length': 5,
#   'food_position': (6, 3),
#   'snake_head': (4, 5)
# }
```

---

## Usage Examples

### Example 1: Random Agent

```python
from snake_env import SnakeEnv
import random

env = SnakeEnv(render_mode='text')

for episode in range(10):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Random action
        action = random.randint(0, 3)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
    
    print(f"Episode {episode + 1}: Reward = {total_reward}")
```

### Example 2: Q-Learning Template

```python
from snake_env import SnakeEnv
import numpy as np

env = SnakeEnv(render_mode='text')

# Initialize Q-table
Q = {}
action_space_size = env.get_action_space_size()

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 1000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Initialize Q-values for new state
        if state not in Q:
            Q[state] = np.zeros(action_space_size)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(0, action_space_size)
        else:
            action = np.argmax(Q[state])
        
        # Take step
        next_state, reward, done = env.step(action)
        
        # Initialize Q-values for next state
        if next_state not in Q:
            Q[next_state] = np.zeros(action_space_size)
        
        # Q-Learning update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error
        
        state = next_state
```

### Example 3: SARSA Template

```python
# SARSA differs from Q-Learning in the update rule

# After taking action and observing next_state:

# Choose next action using epsilon-greedy
if np.random.random() < epsilon:
    next_action = np.random.randint(0, action_space_size)
else:
    next_action = np.argmax(Q[next_state])

# SARSA update (uses next_action that will actually be taken)
td_target = reward + gamma * Q[next_state][next_action]
td_error = td_target - Q[state][action]
Q[state][action] += alpha * td_error

# Update for next iteration
state = next_state
action = next_action  # Use the chosen next action
```

### Example 4: Monte Carlo Template

```python
from snake_env import SnakeEnv
import numpy as np

env = SnakeEnv(render_mode='text')

# Initialize
Q = {}
Returns = {}  # Store returns for each (state, action) pair
action_space_size = env.get_action_space_size()

gamma = 0.99
epsilon = 0.1
num_episodes = 1000

for episode in range(num_episodes):
    # Generate episode
    episode_data = []
    state = env.reset()
    done = False
    
    while not done:
        # Initialize Q-values
        if state not in Q:
            Q[state] = np.zeros(action_space_size)
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(0, action_space_size)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state
    
    # Calculate returns and update Q-values
    G = 0
    for t in range(len(episode_data) - 1, -1, -1):
        state, action, reward = episode_data[t]
        G = gamma * G + reward
        
        # First-visit Monte Carlo
        if (state, action) not in [(x[0], x[1]) for x in episode_data[:t]]:
            if (state, action) not in Returns:
                Returns[(state, action)] = []
            Returns[(state, action)].append(G)
            Q[state][action] = np.mean(Returns[(state, action)])
```

---

## Implementation Notes

### For Your Teammates

1. **State Representation:** The state is a tuple, which can be used as a dictionary key in Python for Q-tables.

2. **Q-Table Storage:** Use a dictionary `Q = {}` where keys are states and values are numpy arrays of Q-values for each action.

3. **Initialization:** Initialize Q-values to 0 for unseen states.

4. **Exploration:** Use epsilon-greedy with ε = 0.1 (or decay from 1.0 to 0.1).

5. **Learning Rate:** Start with α = 0.1 and tune if needed.

6. **Discount Factor:** Use γ = 0.99.

7. **Training:** Run 1000-5000 episodes for convergence.

8. **Evaluation:** After training, test with ε = 0 (pure exploitation).

### Debugging Tips

- **Print episode rewards:** Track if learning is happening
- **Save Q-table periodically:** Use `pickle` or `np.save()`
- **Test with visualization:** Use pygame mode to see behavior
- **Check state counts:** `len(Q)` shows how many states were visited
- **Monitor snake length:** Longer snakes indicate better performance

### Performance Expectations

Random agent typically gets:
- Average reward: -15 to -5 per episode
- Average length: 3 episodes
- Survival: 1-10 steps

Good trained agent should achieve:
- Average reward: 10-50 per episode
- Average length: 6-10 segments
- Survival: 30-100+ steps

---

## File Structure

```
687-Final-Project/
├── snake_env.py              # Your environment (THIS FILE)
├── visualize.py              # Pygame visualization
├── test_env.py              # Test suite and examples
├── train_qlearning.py       # Q-Learning implementation
├── train_sarsa.py           # SARSA implementation
├── train_monte_carlo.py     # Monte Carlo implementation
├── demo_pygame.py           # Demo script
├── requirements.txt         # Dependencies
├── README.md               # Project overview
└── results/                # Plots and results
```

---

## Questions?

If you have questions about the environment, check:
1. `test_env.py` - Complete working examples
2. `demo_pygame.py` - Visual demonstrations
3. This documentation file

For algorithm-specific questions, refer to:
- Sutton & Barto RL Book (Chapter 6 for SARSA/Q-Learning, Chapter 5 for Monte Carlo)
- Course lecture notes

---

**Good luck with your implementations!** 🚀