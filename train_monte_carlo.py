"""
Monte Carlo Control Algorithm for Snake Game

Implementation of First-Visit Monte Carlo Control with reward shaping.
Monte Carlo methods learn from complete episodes rather than bootstrapping.

Author: Rahiq Majeed
Course: COMPSCI 687 - Fall 2025
Algorithm: Monte Carlo Control (Episode-Based Learning)
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeEnv

class RewardShapingWrapper:
    """Adds distance-based reward shaping to guide learning"""
    
    def __init__(self, env):
        self.env = env
        self.prev_distance = None
    
    def reset(self):
        state = self.env.reset()
        head_x, head_y, food_x, food_y, direction = state
        self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
        return state
    
    def step(self, action):
        next_state, reward, done = self.env.step(action)
        
        if not done:
            head_x, head_y, food_x, food_y, direction = next_state
            current_distance = abs(head_x - food_x) + abs(head_y - food_y)
            
            if current_distance < self.prev_distance:
                reward += 1.5
            elif current_distance > self.prev_distance:
                reward -= 0.3
            
            self.prev_distance = current_distance
        
        return next_state, reward, done
    
    def render(self, delay=0):
        return self.env.render(delay)
    
    def close(self):
        self.env.close()
    
    def get_episode_info(self):
        return self.env.get_episode_info()


class MonteCarloAgent:
    """Monte Carlo agent using First-Visit method"""
    
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99996
        
        self.Q = {}
        self.returns = {}  # Store returns for each (state, action) pair
        self.optimistic_value = 5.0
        
    def get_q_values(self, state):
        if state not in self.Q:
            self.Q[state] = np.ones(4) * self.optimistic_value
        return self.Q[state]
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.get_q_values(state))
    
    def update_from_episode(self, episode):
        """
        Update Q-values from complete episode using First-Visit Monte Carlo
        
        Args:
            episode: List of (state, action, reward) tuples
        """
        # Calculate returns for each time step
        G = 0
        visited_state_actions = set()
        
        # Work backwards through episode
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            # Update return
            G = self.gamma * G + reward
            
            # First-visit: only update if this is first time we saw (state, action)
            state_action = (state, action)
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                
                # Store return
                if state_action not in self.returns:
                    self.returns[state_action] = []
                self.returns[state_action].append(G)
                
                # Ensure state exists in Q-table before updating
                self.get_q_values(state)
                
                # Update Q-value as average of returns
                self.Q[state][action] = np.mean(self.returns[state_action])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_greedy_action(self, state):
        return np.argmax(self.get_q_values(state))


def train_monte_carlo(num_episodes=200000, print_every=5000):
    """Train Monte Carlo agent"""
    
    print("="*70)
    print("MONTE CARLO TRAINING")
    print("="*70)
    print(f"Training for {num_episodes:,} episodes")
    print("="*70 + "\n")
    
    env = RewardShapingWrapper(SnakeEnv(render_mode='text'))
    agent = MonteCarloAgent()
    
    best_reward = -float('inf')
    best_length = 0
    
    for episode_num in range(num_episodes):
        state = env.reset()
        done = False
        episode_data = []  # Store (state, action, reward) for this episode
        episode_reward = 0
        
        # Generate episode
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            episode_data.append((state, action, reward))
            episode_reward += reward
            
            state = next_state
        
        # Update Q-values from complete episode
        agent.update_from_episode(episode_data)
        
        info = env.get_episode_info()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_length = info['snake_length']
        
        agent.decay_epsilon()
        
        if (episode_num + 1) % print_every == 0:
            print(f"Episode {episode_num+1:7,}/{num_episodes:,} completed")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best reward: {best_reward:.1f}")
    print(f"Best length: {best_length}")
    print(f"States explored: {len(agent.Q):,}")
    print(f"State-action pairs with returns: {len(agent.returns):,}")
    print("="*70 + "\n")
    
    return agent


def evaluate_agent(agent, num_episodes=20):
    """Evaluate trained agent and create results visualization"""
    
    print("="*70)
    print("EVALUATING TRAINED AGENT")
    print("="*70)
    
    env = RewardShapingWrapper(SnakeEnv(render_mode='pygame', cell_size=70))
    
    rewards = []
    lengths = []
    steps_list = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        env.render(delay=500)
        
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            env.render(delay=80)
        
        info = env.get_episode_info()
        rewards.append(episode_reward)
        lengths.append(info['snake_length'])
        steps_list.append(info['steps'])
        
        print(f"Episode {ep+1:2d}: Reward={episode_reward:7.1f}, Length={info['snake_length']:2d}, Steps={info['steps']:3d}")
    
    env.close()
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print(f"Best Reward: {max(rewards):.1f}")
    print(f"Best Length: {max(lengths)}")
    print("="*70 + "\n")
    
    # Create results graph
    plot_results(rewards, lengths, steps_list)
    
    return rewards, lengths, steps_list


def plot_results(rewards, lengths, steps):
    """Create single clean results graph"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    episodes = range(1, len(rewards) + 1)
    
    # Plot reward as bars
    bars = ax.bar(episodes, rewards, color='darkorange', alpha=0.7, label='Episode Reward')
    
    # Add average line
    avg_reward = np.mean(rewards)
    ax.axhline(y=avg_reward, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_reward:.1f}')
    
    # Formatting
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=13, fontweight='bold')
    ax.set_title('Monte Carlo Evaluation Results (20 Episodes)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    stats_text = (
        f"Statistics:\n"
        f"Avg Reward: {np.mean(rewards):.1f}\n"
        f"Best Reward: {max(rewards):.1f}\n"
        f"Avg Length: {np.mean(lengths):.1f}\n"
        f"Best Length: {max(lengths)}\n"
        f"Avg Steps: {np.mean(steps):.1f}"
    )
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/monte_carlo_results.png', dpi=150, bbox_inches='tight')
    print("✓ Results saved to results/monte_carlo_results.png\n")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MONTE CARLO FOR SNAKE GAME")
    print("="*70 + "\n")
    
    # Train
    agent = train_monte_carlo(num_episodes=200000, print_every=5000)
    
    # Evaluate and save results
    evaluate_agent(agent, num_episodes=20)
    
    print("="*70)
    print("COMPLETE")
    print("="*70 + "\n")