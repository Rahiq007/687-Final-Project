"""
Text-Based Demo for Snake RL Environment

This version uses text rendering for FAST demonstration without pygame window.
Use this when you want to quickly test the environment without visualization. (will be deleted later)

Author: Rahiq Majeed
Course: COMPSCI 687 - Fall 2025
"""

import random
from snake_env import SnakeEnv

def run_baseline_agent(num_episodes=10):
    """
    Run a baseline random agent (no learning)
    This shows what performance looks like WITHOUT any RL algorithm
    """
    print("="*70)
    print("BASELINE: Random Agent (No Learning)")
    print("="*70)
    print("\nThis agent takes random actions with no learning.")
    print("Use this as a baseline to compare against your RL algorithms.\n")
    
    env = SnakeEnv(render_mode='text')
    
    episode_rewards = []
    episode_lengths = []
    episode_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Random action (baseline - no learning)
            action = random.randint(0, 3)
            state, reward, done = env.step(action)
        
        # Collect statistics
        info = env.get_episode_info()
        episode_rewards.append(info['total_reward'])
        episode_lengths.append(info['snake_length'])
        episode_steps.append(info['steps'])
        
        print(f"Episode {episode + 1:3d}: Reward = {info['total_reward']:6.1f}, "
              f"Length = {info['snake_length']}, Steps = {info['steps']}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("BASELINE STATISTICS:")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Average Snake Length: {sum(episode_lengths)/len(episode_lengths):.2f}")
    print(f"Average Steps: {sum(episode_steps)/len(episode_steps):.2f}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Worst Reward: {min(episode_rewards):.2f}")
    print("="*70)
    
    return {
        'avg_reward': sum(episode_rewards)/len(episode_rewards),
        'avg_length': sum(episode_lengths)/len(episode_lengths),
        'avg_steps': sum(episode_steps)/len(episode_steps)
    }


def run_directional_agent(num_episodes=10):
    """
    Run a simple heuristic agent that moves toward food
    This is better than random but not optimal
    """
    print("\n" + "="*70)
    print("HEURISTIC: Directional Agent (Simple Strategy)")
    print("="*70)
    print("\nThis agent always moves toward the food.")
    print("It's smarter than random but not optimal.\n")
    
    env = SnakeEnv(render_mode='text')
    
    episode_rewards = []
    episode_lengths = []
    episode_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            head_x, head_y, food_x, food_y, current_dir = state
            
            # Simple heuristic: move toward food
            dx = food_x - head_x
            dy = food_y - head_y
            
            if abs(dx) > abs(dy):
                action = env.RIGHT if dx > 0 else env.LEFT
            else:
                action = env.DOWN if dy > 0 else env.UP
            
            state, reward, done = env.step(action)
        
        # Collect statistics
        info = env.get_episode_info()
        episode_rewards.append(info['total_reward'])
        episode_lengths.append(info['snake_length'])
        episode_steps.append(info['steps'])
        
        print(f"Episode {episode + 1:3d}: Reward = {info['total_reward']:6.1f}, "
              f"Length = {info['snake_length']}, Steps = {info['steps']}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("HEURISTIC STATISTICS:")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Average Snake Length: {sum(episode_lengths)/len(episode_lengths):.2f}")
    print(f"Average Steps: {sum(episode_steps)/len(episode_steps):.2f}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Worst Reward: {min(episode_rewards):.2f}")
    print("="*70)
    
    return {
        'avg_reward': sum(episode_rewards)/len(episode_rewards),
        'avg_length': sum(episode_lengths)/len(episode_lengths),
        'avg_steps': sum(episode_steps)/len(episode_steps)
    }


def compare_agents():
    """
    Compare baseline random agent vs heuristic agent
    This gives you baseline numbers to compare your RL algorithms against
    """
    print("\n" + "="*70)
    print("RUNNING COMPARISON: BASELINE vs HEURISTIC")
    print("="*70)
    
    # Run both agents
    baseline_stats = run_baseline_agent(num_episodes=20)
    heuristic_stats = run_directional_agent(num_episodes=20)
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY:")
    print("="*70)
    print(f"{'Metric':<20} {'Random':<15} {'Heuristic':<15} {'Improvement'}")
    print("-"*70)
    print(f"{'Avg Reward':<20} {baseline_stats['avg_reward']:<15.2f} "
          f"{heuristic_stats['avg_reward']:<15.2f} "
          f"{(heuristic_stats['avg_reward'] - baseline_stats['avg_reward']):.2f}")
    print(f"{'Avg Length':<20} {baseline_stats['avg_length']:<15.2f} "
          f"{heuristic_stats['avg_length']:<15.2f} "
          f"{(heuristic_stats['avg_length'] - baseline_stats['avg_length']):.2f}")
    print(f"{'Avg Steps':<20} {baseline_stats['avg_steps']:<15.2f} "
          f"{heuristic_stats['avg_steps']:<15.2f} "
          f"{(heuristic_stats['avg_steps'] - baseline_stats['avg_steps']):.2f}")
    print("="*70)
    
    print("\n💡 Use these numbers as BASELINE in your report!")
    print("   Your RL algorithms should beat the heuristic agent.\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SNAKE RL ENVIRONMENT - TEXT-BASED DEMO")
    print("="*70)
    print("\nThis demo runs quickly without visualization.")
    print("Use this to get baseline numbers for your report.\n")
    
    # Run comparison
    compare_agents()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nFor visual demonstrations, run: python demo_pygame.py")
    print("="*70 + "\n")