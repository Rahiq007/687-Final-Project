# Snake Game Reinforcement Learning Project Report

## 1. Problem Description and MDP Formulation

### Problem Overview
The project addresses the classic game of **Snake**, modeled as a Reinforcement Learning problem. The objective is to control a snake on a grid to eat food, grow in length, and avoid collisions with walls, obstacles, and itself.

### MDP components
The problem is formally defined as a Markov Decision Process (MDP) tuple $\langle S, A, R, P, \gamma \rangle$:

#### 1. State Space ($S$)
The state is a compact representation of the environment designed to capture essential information while keeping the state space manageable (dimensionality reduction). 
The state is a tuple $s = (d_{food}, s_{surround}, c_{len})$, where:
*   **Food Direction ($d_{food}$)**: Represents the relative direction of the food from the snake's head. Discretized into **8 sectors** (N, NE, E, SE, S, SW, W, NW).
*   **Surroundings ($s_{surround}$)**: encodes the safety and utility of the immediate adjacent cells (Up, Down, Left, Right). Each direction is assigned one of **4 discrete levels**:
    *   `0` (Death): Immediate collision (Wall, Body, Obstacle).
    *   `1` (Trap): Move leads to a confined space smaller than the snake's length.
    *   `2` (Safe & Neutral): Safe move, but does not decrease distance to food.
    *   `3` (Safe & Closer): Safe move that decreases Manhattan distance to food.
    *   The total surround state is encoded as a base-4 integer: $\sum_{i=0}^{3} level_i \cdot 4^i$. (Total $4^4 = 256$ combinations).
*   **Length Category ($c_{len}$)**: Categorizes the snake's current length to allow different strategies for different stages of the game.
    *   `0` (Small): Length < 8
    *   `1` (Medium): 8 $\le$ Length < 20
    *   `2` (Large): Length $\ge$ 20

**Total State Space Size:** $8 \times 256 \times 3 = 6,144$ states.

#### 2. Action Space ($A$)
The action space is discrete with 4 possible moves:
*   $A = \{ \text{UP (0), DOWN (1), LEFT (2), RIGHT (3)} \}$

#### 3. Reward Function ($R$)
The reward function is designed to guide the agent towards food while strictly penalizing inefficiency and death.
*   **Step Reward**:
    *   Moving **Closer** to food: $-0.1$
    *   Moving **Further** from food: $-0.5$
*   **Food Reward**: $+1.0 + (0.5 \times \text{SnakeLength})$ (Scales with growth)
*   **Death Penalty**: $-3.0 - (1.0 \times \text{SnakeLength})$ (Scales with size to discourage losing long snakes)
*   **Win Bonus**: $+200$ (for clearing the board)

#### 4. Transition Dynamics ($P$)
*   The environment is **deterministic**. Given a state $s$ and action $a$, the snake moves to the adjacent cell in direction $a$.
*   If the cell contains food, the snake grows (tail remains).
*   If the cell is empty, the snake moves (tail retracts).
*   If the cell is a wall/body/obstacle, the episode ends (terminal state).
*   Food respawns randomly in an empty cell.

#### 5. Discount Factor ($\gamma$)
*   $\gamma = 0.99$. This encourages the agent to value long-term survival and future food rewards significantly.

## 2. Implementation Details

### Language and Tools
*   **Language**: Python 3
*   **Libraries**: `numpy` (matrix operations), `pygame` (rendering), `matplotlib` (plotting).
*   **Architecture**:
    *   `SnakeEnvV9`: Custom environment class implementing the MDP logic.
    *   `SarsaAgentV9`: Agent class implementing the tabular SARSA algorithm.

### Assumptions
1.  **Stationarity**: The environment rules do not change over time.
2.  **Full Observability assumption within State**: While the full grid is not passed to the Q-table, we assume the engineered state tuple (local view + food direction) provides sufficient statistics to make optimal local decisions.
3.  **Discrete Time**: Decisions are made at fixed time steps. (The game waits for agent input).

## 3. RL Algorithm and Hyperparameters

### Algorithm: SARSA
We used **SARSA (State-Action-Reward-State-Action)**, an on-policy TD control algorithm.
Update rule:
$$Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$$

### Hyperparameters
| Parameter | Symbol | Value | Description |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | $\alpha$ | `0.1` | Determines how much new information overrides old information. |
| **Discount Factor** | $\gamma$ | `0.99` | Weights future rewards; high value for long-term planning. |
| **Initial Epsilon** | $\epsilon_{start}$ | `1.0` | 100% exploration at the start. |
| **Minimum Epsilon** | $\epsilon_{min}$ | `0.01` | 1% random noise for robustness in late training. |
| **Epsilon Decay Strategy** | - | **Linear Annealing** | Epsilon decreases linearly from 1.0 to 0.01 over the first **70%** of episodes. |
| **Total Episodes** | $N$ | `30,000` | Duration of training. |

### Hyperparameter Optimization
*   **Epsilon Schedule**: A standard exponential decay was found to be too aggressive, leading to premature convergence to suboptimal policies (loops). A linear schedule extending over 70% of the training time proved more effective for exploring the complex state space.
*   **Reward Shaping**: The penalty for simply existing (step penalty) was split into "closer" vs "further" to provide a denser reward signal, which significantly accelerated the initial learning of food-seeking behavior.

## 4. Performance Analysis
*(Refer to generated plots in `lucas_dev/plots` for visual learning curves)*

*   **Learning Curve**: The agent typically shows a slow initial improvements as it explores the grid, followed by a rapid increase in average length around episode 2,000-5,0000.
*   **Stability**: The linear decay schedule ensures that once the agent learns a robust policy, performance stabilizes with high average snake lengths.
*   **Convergence**: By the end of training (30k episodes), the agent consistently achieves high scores, demonstrating that the Q-table has converged to an effective policy for the given state representation.
