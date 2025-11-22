"""
Example: Training with Pygame Visualization

This shows how to train your RL algorithm while WATCHING it learn in real-time!

This is a TEMPLATE to follow. (will be deleted later)
You both should adapt this for Q-Learning, SARSA, and Monte Carlo. 

Author: Rahiq Majeed
Course: COMPSCI 687 - Fall 2025
"""

import numpy as np
from snake_env import SnakeEnv
import matplotlib.pyplot as plt

def train_with_visualization(num_episodes=100, render_every=1, delay=50):
    """
    Train an agent with pygame visualization
    
    Args:
        num_episodes: Total episodes to train (default: 100)
        render_every: Render every Nth episode (1=all, 10=every 10th)
        delay: Milliseconds between steps when rendering (50=fast, 200=slow)
    
    TIPS:
    - Use render_every=1 for first 50 episodes to watch learning
    - Use render_every=10 for episodes 50-500 to speed up
    - Use render_every=100 for episodes 500+ to speed up more
    """
    print("="*70)
    print("TRAINING WITH VISUALIZATION")
    print("="*70)
    print(f"Training for {num_episodes} episodes")
    print(f"Rendering every {render_every} episode(s)")
    print(f"Animation delay: {delay}ms")
    print("="*70)
    
    # Create environment (pygame mode by default)
    env = SnakeEnv(render_mode='pygame', cell_size=60)
    
    # Initialize Q-table (this is a simple example - use your actual algorithm!)
    Q = {}
    action_space_size = env.get_action_space_size()
    
    # Hyperparameters
    alpha = 0.1      # Learning rate
    gamma = 0.99     # Discount factor
    epsilon = 0.5    # Exploration rate (start higher to explore)
    
    # Track statistics
    episode_rewards = []
    episode_lengths = []
    episode_steps = []
    
    # TRAINING LOOP
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        # Decide if we render this episode
        should_render = (episode % render_every == 0)
        
        if should_render:
            env.render(delay=300)  # Show initial state a bit longer
        
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
            
            # Render if this is a rendered episode
            if should_render:
                env.render(delay=delay)
            
            # Initialize Q-values for next state
            if next_state not in Q:
                Q[next_state] = np.zeros(action_space_size)
            
            # Q-LEARNING UPDATE (you can change this to SARSA or Monte Carlo)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = next_state
        
        # Collect episode statistics
        info = env.get_episode_info()
        episode_rewards.append(info['total_reward'])
        episode_lengths.append(info['snake_length'])
        episode_steps.append(info['steps'])
        
        # Print progress every episode
        print(f"Episode {episode + 1:4d}: "
              f"Reward = {info['total_reward']:6.1f}, "
              f"Length = {info['snake_length']:2d}, "
              f"Steps = {info['steps']:3d}, "
              f"Q-states = {len(Q):5d}")
        
        # Decay epsilon (explore less over time)
        epsilon = max(0.1, epsilon * 0.998)  
    
    # Clean up
    env.close()
    
    # PRINT FINAL STATISTICS
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Episodes Trained: {num_episodes}")
    print(f"States Visited: {len(Q)}")
    print("\nFINAL PERFORMANCE (Last 20 episodes):")
    print(f"  Average Reward: {np.mean(episode_rewards[-20:]):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths[-20:]):.2f}")
    print(f"  Average Steps: {np.mean(episode_steps[-20:]):.2f}")
    print(f"  Best Reward: {max(episode_rewards[-20:]):.2f}")
    print("\nOVERALL PERFORMANCE:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Best Reward Ever: {max(episode_rewards):.2f}")
    print(f"  Worst Reward Ever: {min(episode_rewards):.2f}")
    print("="*70)
    
    # Plot learning curve
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    window = 20
    moving_avg = [np.mean(episode_rewards[max(0, i-window):i+1]) for i in range(len(episode_rewards))]
    plt.plot(moving_avg, linewidth=2, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve: Reward over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Snake length
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, alpha=0.3, label='Snake Length')
    moving_avg_len = [np.mean(episode_lengths[max(0, i-window):i+1]) for i in range(len(episode_lengths))]
    plt.plot(moving_avg_len, linewidth=2, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Snake Length')
    plt.title('Learning Curve: Snake Length over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    print(f"\n📊 Learning curves saved to: results/training_curves.png")
    plt.show()
    
    return Q, episode_rewards, episode_lengths


def demonstrate_trained_agent(Q, num_episodes=5):
    """
    Demonstrate the trained agent with NO exploration
    
    Args:
        Q: Trained Q-table
        num_episodes: Number of episodes to show (default: 5)
    """
    print("\n" + "="*70)
    print("DEMONSTRATING TRAINED AGENT")
    print("="*70)
    print(f"Showing {num_episodes} episodes with learned policy (no exploration)")
    print("="*70 + "\n")
    
    env = SnakeEnv(render_mode='pygame', cell_size=70)
    action_space_size = env.get_action_space_size()
    
    demo_rewards = []
    demo_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        env.render(delay=500)  # Show initial state
        
        while not done:
            # Use learned policy (NO exploration)
            if state in Q:
                action = np.argmax(Q[state])
            else:
                # Unseen state - use default
                action = np.random.randint(0, action_space_size)
            
            state, reward, done = env.step(action)
            env.render(delay=150)  # Slower so you can watch
        
        info = env.get_episode_info()
        demo_rewards.append(info['total_reward'])
        demo_lengths.append(info['snake_length'])
        
        print(f"Demo Episode {episode + 1}: "
              f"Reward = {info['total_reward']:6.1f}, "
              f"Length = {info['snake_length']:2d}, "
              f"Steps = {info['steps']:3d}")
    
    env.close()
    
    # Print demo statistics
    print("\n" + "="*70)
    print("DEMONSTRATION STATISTICS:")
    print("="*70)
    print(f"Average Reward: {np.mean(demo_rewards):.2f}")
    print(f"Average Length: {np.mean(demo_lengths):.2f}")
    print(f"Best Reward: {max(demo_rewards):.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE: TRAINING WITH VISUALIZATION")
    print("="*70)
    print("\nThis example shows how to:")
    print("  1. Train an RL agent while WATCHING it learn")
    print("  2. Get terminal output with statistics")
    print("  3. Generate learning curves")
    print("  4. Demonstrate the final trained policy")
    print("\nAdapt this template for your Q-Learning, SARSA, and Monte Carlo!")
    print("="*70 + "\n")
    
    # Train (you can watch it learn!)
    # Start with 50 episodes to see how it works
    Q, rewards, lengths = train_with_visualization(
        num_episodes=200,      # Start small to see learning
        render_every=1,       # Render every episode
        delay=150              # Fast animation (50ms per step)
    )
    
    # Demonstrate trained agent
    demonstrate_trained_agent(Q, num_episodes=5)
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE!")
    print("="*70)
    print("\n💡 FOR YOUR ACTUAL TRAINING:")
    print("   - Train for 1000+ episodes")
    print("   - Use render_every=10 or 50 to speed up")
    print("   - Save your Q-table with pickle")
    print("   - Generate comparison plots")
    print("="*70 + "\n")