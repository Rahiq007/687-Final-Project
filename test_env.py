"""
Test Script for Snake RL Environment

This script demonstrates how to use the Snake environment for RL algorithms.
Your teammates should use this as a reference for implementing SARSA, Q-Learning, and Monte Carlo.

Author: [Your name]
Course: COMPSCI 687 - Fall 2025
"""

import numpy as np
from snake_env import SnakeEnv
import random

def test_basic_interface():
    """Test 1: Basic environment interface"""
    print("="*70)
    print("TEST 1: Basic Environment Interface")
    print("="*70)
    
    # Create environment
    env = SnakeEnv(render_mode='text')
    
    # Get environment information
    print(f"\nAction Space Size: {env.get_action_space_size()}")
    print(f"State Space Size: {env.get_state_space_size()}")
    print(f"Grid Size: {env.get_grid_size()}x{env.get_grid_size()}")
    print(f"Obstacles: {env.get_obstacles()}")
    print(f"\nState Bounds: {env.get_state_bounds()}")
    
    # Reset and get initial state
    state = env.reset()
    print(f"\nInitial State: {state}")
    print(f"  head_x={state[0]}, head_y={state[1]}")
    print(f"  food_x={state[2]}, food_y={state[3]}")
    print(f"  direction={state[4]}")
    
    print("\n✅ Test 1 Passed!")


def test_episode_execution():
    """Test 2: Running a complete episode"""
    print("\n" + "="*70)
    print("TEST 2: Episode Execution")
    print("="*70)
    
    env = SnakeEnv(render_mode='text')
    
    # Run one episode with random actions
    state = env.reset()
    done = False
    step = 0
    
    print("\nRunning episode with random actions...")
    
    while not done and step < 20:
        # Choose random action
        action = random.randint(0, 3)
        
        # Take step
        next_state, reward, done = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
        print(f"  Reward: {reward}")
        print(f"  Next State: {next_state}")
        print(f"  Done: {done}")
        
        state = next_state
        step += 1
        
        if done:
            print(f"\n  Episode ended after {step} steps!")
            break
    
    # Get episode information
    info = env.get_episode_info()
    print(f"\nEpisode Info:")
    print(f"  Steps: {info['steps']}")
    print(f"  Total Reward: {info['total_reward']}")
    print(f"  Final Snake Length: {info['snake_length']}")
    
    print("\n✅ Test 2 Passed!")


def test_state_collection():
    """Test 3: Collecting state-action-reward data for RL"""
    print("\n" + "="*70)
    print("TEST 3: Data Collection for RL")
    print("="*70)
    
    env = SnakeEnv(render_mode='text')
    
    # Data structure to collect experiences
    experiences = []
    
    print("\nRunning 5 episodes and collecting data...")
    
    for episode in range(5):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose random action
            action = random.randint(0, 3)
            
            # Take step
            next_state, reward, done = env.step(action)
            
            # Store experience (s, a, r, s', done)
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }
            experiences.append(experience)
            
            episode_reward += reward
            state = next_state
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward}")
    
    print(f"\nCollected {len(experiences)} total experiences")
    print(f"First experience: {experiences[0]}")
    
    print("\n✅ Test 3 Passed!")


def test_q_table_structure():
    """Test 4: Q-table structure example"""
    print("\n" + "="*70)
    print("TEST 4: Q-Table Structure for Tabular Methods")
    print("="*70)
    
    env = SnakeEnv(render_mode='text')
    
    # Get dimensions
    action_space_size = env.get_action_space_size()
    grid_size = env.get_grid_size()
    
    print(f"\nFor Q-Learning/SARSA, you need a Q-table with dimensions:")
    print(f"  State space: {grid_size} x {grid_size} x {grid_size} x {grid_size} x 4")
    print(f"  Action space: {action_space_size}")
    
    # Example Q-table initialization using dictionary
    Q = {}
    
    # Run one episode and initialize Q-values for visited states
    state = env.reset()
    done = False
    
    print("\nInitializing Q-values for states encountered in one episode...")
    
    step = 0
    while not done and step < 30:
        # Initialize Q-values for this state if not seen before
        if state not in Q:
            Q[state] = np.zeros(action_space_size)
        
        # Choose random action
        action = random.randint(0, 3)
        
        # Take step
        next_state, reward, done = env.step(action)
        
        state = next_state
        step += 1
    
    print(f"Q-table now has {len(Q)} state entries")
    print(f"Example Q-values for a state: {list(Q.values())[0]}")
    
    print("\n✅ Test 4 Passed!")


def example_training_loop():
    """Example: Training loop structure"""
    print("\n" + "="*70)
    print("EXAMPLE: Training Loop Structure")
    print("="*70)
    
    print("""
# Example training loop for your RL algorithms:

from snake_env import SnakeEnv
import numpy as np

# Create environment
env = SnakeEnv(render_mode='text')

# Initialize Q-table (or policy)
Q = {}
action_space_size = env.get_action_space_size()

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 0.1        # Exploration rate
num_episodes = 1000

# Training
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Initialize Q-values for new states
        if state not in Q:
            Q[state] = np.zeros(action_space_size)
        
        # Choose action (epsilon-greedy)
        if np.random.random() < epsilon:
            action = np.random.randint(0, action_space_size)  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit
        
        # Take step
        next_state, reward, done = env.step(action)
        
        # Initialize Q-values for next state
        if next_state not in Q:
            Q[next_state] = np.zeros(action_space_size)
        
        # Q-Learning Update:
        # Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
        # SARSA Update (choose next action first):
        # next_action = choose_action(next_state)
        # Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        state = next_state
    
    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}")

# Save trained Q-table or policy
# np.save('q_table.npy', Q)

# Test trained agent with visualization
env_test = SnakeEnv(render_mode='pygame')
# ... run episodes using learned Q-values ...
env_test.close()
""")
    
    print("\n✅ Example Provided!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SNAKE RL ENVIRONMENT - TEST SUITE")
    print("="*70)
    print("\nThis script tests all functionality needed for RL algorithms.")
    print("Your teammates can use this as a reference!\n")
    
    # Run all tests
    test_basic_interface()
    test_episode_execution()
    test_state_collection()
    test_q_table_structure()
    example_training_loop()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print("\nThe environment is ready for RL algorithm implementation!")
    print("\nNext steps for your teammates:")
    print("  1. Implement Q-Learning in train_qlearning.py")
    print("  2. Implement SARSA in train_sarsa.py")
    print("  3. Implement Monte Carlo in train_monte_carlo.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()