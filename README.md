# Snake Game - Reinforcement Learning Final Project

**Course**: COMPSCI 687 - Fall 2025  
**Group Members**: [Add your names here]

## Project Overview

This project implements a Snake game environment and trains three RL algorithms to play it:
1. SARSA
2. Monte Carlo Control
3. Q-Learning

## Environment Details

- **Grid Size**: 8x8
- **Actions**: UP (0), DOWN (1), LEFT (2), RIGHT (3)
- **State Representation**: (head_x, head_y, food_x, food_y, direction)
- **Rewards**:
  - +10 for eating food
  - -20 for death (wall, self-collision, obstacle)
  - -1 for each step
- **Obstacles**: 3-5 fixed obstacles

## File Structure

- `snake_env.py` - Snake environment implementation
- `test_env.py` - Environment testing script
- `train_sarsa.py` - SARSA algorithm implementation
- `train_monte_carlo.py` - Monte Carlo Control implementation
- `train_qlearning.py` - Q-Learning algorithm implementation
- `visualize.py` - Visualization utilities
- `results/` - Training results and plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Test the Environment
```bash
python test_env.py
```

### Train Algorithms
```bash
python train_sarsa.py
python train_monte_carlo.py
python train_qlearning.py
```

## TODO
- [ ] Complete environment implementation
- [ ] Implement SARSA
- [ ] Implement Monte Carlo Control
- [ ] Implement Q-Learning
- [ ] Generate learning curves
- [ ] Write final report