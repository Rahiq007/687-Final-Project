"""
Demo script for Snake RL Environment with Pygame Visualization

This script demonstrates:
1. How to use the pygame visualization
2. Random agent playing the game
3. How the visualization looks during gameplay

To run the demo, execute this script. You can choose between a random agent or a simple directional agent.
The random agent takes random actions, while the directional agent tries to move toward the food.
The demo runs for a specified number of episodes, rendering the game state in a Pygame window.
To stop the demo, simply close the Pygame window.
This file is just to test the Pygame works as expected (will be deleted later).
"""

import random
import time
from snake_env import SnakeEnv

def run_random_agent_demo(num_episodes=5, max_steps_per_episode=150):
    """
    Run a random agent that takes random actions
    
    Args:
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
    """
    print("="*70)
    print("SNAKE RL ENVIRONMENT - RANDOM AGENT DEMO")
    print("="*70)
    print("\nThis demo shows a random agent playing the game.")
    print("The agent will take random actions at each step.")
    print("\nClose the window to stop the demo.")
    print("="*70)
    
    # Create environment with pygame visualization
    env = SnakeEnv(render_mode='pygame', cell_size=70)
    
    try:
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset environment
            state = env.reset()
            env.render(delay=500)  # Show initial state for 0.5 seconds
            
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps_per_episode:
                # Random action
                action = random.randint(0, 3)
                
                # Take action
                next_state, reward, done = env.step(action)
                total_reward += reward
                steps += 1
                
                # Render with delay (adjust delay to control speed)
                env.render(delay=100)  # 100ms between moves (faster)
                
                state = next_state
            
            # Show episode results
            print(f"Episode {episode + 1} finished!")
            print(f"  Steps: {steps}")
            print(f"  Total Reward: {total_reward}")
            print(f"  Final Snake Length: {len(env.snake)}")
            
            # Wait before next episode
            if episode < num_episodes - 1:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)
    
    finally:
        # Clean up
        env.close()
        print("\n" + "="*70)
        print("Demo completed!")
        print("="*70)


def run_directional_agent_demo(num_episodes=5):
    """
    Run an agent that moves toward the food
    
    This is a simple heuristic agent that moves in the direction of the food.
    It's not perfect but shows more interesting behavior than random.
    """
    print("="*70)
    print("SNAKE RL ENVIRONMENT - DIRECTIONAL AGENT DEMO")
    print("="*70)
    print("\nThis demo shows an agent that tries to move toward the food.")
    print("\nClose the window to stop the demo.")
    print("="*70)
    
    # Create environment with pygame visualization
    env = SnakeEnv(render_mode='pygame', cell_size=70)
    
    try:
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset environment
            state = env.reset()
            env.render(delay=500)
            
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 200:
                # Get current position and food position
                head_x, head_y, food_x, food_y, current_dir = state
                
                # Choose action to move toward food
                # This is a simple heuristic - not optimal but better than random
                
                # Calculate differences
                dx = food_x - head_x
                dy = food_y - head_y
                
                # Prioritize horizontal or vertical movement
                if abs(dx) > abs(dy):
                    # Move horizontally
                    if dx > 0:
                        action = env.RIGHT
                    else:
                        action = env.LEFT
                else:
                    # Move vertically
                    if dy > 0:
                        action = env.DOWN
                    else:
                        action = env.UP
                
                # Take action
                next_state, reward, done = env.step(action)
                total_reward += reward
                steps += 1
                
                # Render (faster than random agent)
                env.render(delay=100)
                
                state = next_state
            
            # Show episode results
            print(f"Episode {episode + 1} finished!")
            print(f"  Steps: {steps}")
            print(f"  Total Reward: {total_reward}")
            print(f"  Final Snake Length: {len(env.snake)}")
            
            # Wait before next episode
            if episode < num_episodes - 1:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)
    
    finally:
        # Clean up
        env.close()
        print("\n" + "="*70)
        print("Demo completed!")
        print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHOOSE A DEMO:")
    print("="*70)
    print("1. Random Agent (takes random actions) - 5 episodes")
    print("2. Directional Agent (moves toward food) - 5 episodes")
    print("="*70)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_random_agent_demo(num_episodes=5, max_steps_per_episode=150)
    elif choice == "2":
        run_directional_agent_demo(num_episodes=5)
    else:
        print("Invalid choice. Running random agent demo...")
        run_random_agent_demo(num_episodes=5, max_steps_per_episode=150)