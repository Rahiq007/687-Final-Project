"""
Complete Comparison of Q-Learning, SARSA, and Monte Carlo

This script trains all three algorithms and creates comparison plots.

Author: Rahiq Majeed
Course: COMPSCI 687 - Fall 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeEnv
import time

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
    
    def get_episode_info(self):
        return self.env.get_episode_info()


class QLearningAgent:
    """Q-Learning agent"""
    
    def __init__(self):
        self.alpha_start = 0.3
        self.alpha_min = 0.05
        self.alpha = self.alpha_start
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99996
        self.Q = {}
        self.optimistic_value = 5.0
    
    def get_q_values(self, state):
        if state not in self.Q:
            self.Q[state] = np.ones(4) * self.optimistic_value
        return self.Q[state]
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q_values(state)[action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.get_q_values(next_state))
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_alpha(self, episode, total_episodes):
        progress = episode / total_episodes
        self.alpha = self.alpha_min + (self.alpha_start - self.alpha_min) * (1 - progress)


class SARSAAgent:
    """SARSA agent"""
    
    def __init__(self):
        self.alpha_start = 0.3
        self.alpha_min = 0.05
        self.alpha = self.alpha_start
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99996
        self.Q = {}
        self.optimistic_value = 5.0
    
    def get_q_values(self, state):
        if state not in self.Q:
            self.Q[state] = np.ones(4) * self.optimistic_value
        return self.Q[state]
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.get_q_values(state)[action]
        if done:
            target = reward
        else:
            next_q = self.get_q_values(next_state)[next_action]
            target = reward + self.gamma * next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_alpha(self, episode, total_episodes):
        progress = episode / total_episodes
        self.alpha = self.alpha_min + (self.alpha_start - self.alpha_min) * (1 - progress)


class MonteCarloAgent:
    """Monte Carlo agent"""
    
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99996
        self.Q = {}
        self.returns = {}
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
        G = 0
        visited_state_actions = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            state_action = (state, action)
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                if state_action not in self.returns:
                    self.returns[state_action] = []
                self.returns[state_action].append(G)
                self.get_q_values(state)
                self.Q[state][action] = np.mean(self.returns[state_action])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_all_algorithms(num_episodes=200000, window=500):
    """Train all three algorithms and track progress"""
    
    print("="*70)
    print("TRAINING ALL THREE ALGORITHMS")
    print("="*70)
    print(f"Episodes: {num_episodes:,}")
    print(f"This will take approximately 2-3 minutes")
    print("="*70 + "\n")
    
    # Create agents
    q_agent = QLearningAgent()
    sarsa_agent = SARSAAgent()
    mc_agent = MonteCarloAgent()
    
    # Track statistics
    q_rewards = []
    sarsa_rewards = []
    mc_rewards = []
    
    q_avg_rewards = []
    sarsa_avg_rewards = []
    mc_avg_rewards = []
    
    episodes_list = []
    
    start_time = time.time()
    
    print("Training Q-Learning...")
    env = RewardShapingWrapper(SnakeEnv(render_mode='text'))
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = q_agent.choose_action(state)
            next_state, reward, done = env.step(action)
            q_agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        
        q_rewards.append(episode_reward)
        q_agent.decay_epsilon()
        q_agent.update_alpha(ep, num_episodes)
        
        if (ep + 1) % window == 0:
            avg = np.mean(q_rewards[-window:])
            q_avg_rewards.append(avg)
            episodes_list.append(ep + 1)
            print(f"  Episode {ep+1:6,}: Avg Reward = {avg:6.1f}")
    
    print("\nTraining SARSA...")
    env = RewardShapingWrapper(SnakeEnv(render_mode='text'))
    for ep in range(num_episodes):
        state = env.reset()
        action = sarsa_agent.choose_action(state)
        done = False
        episode_reward = 0
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = sarsa_agent.choose_action(next_state)
            sarsa_agent.update(state, action, reward, next_state, next_action, done)
            episode_reward += reward
            state = next_state
            action = next_action
        
        sarsa_rewards.append(episode_reward)
        sarsa_agent.decay_epsilon()
        sarsa_agent.update_alpha(ep, num_episodes)
        
        if (ep + 1) % window == 0:
            avg = np.mean(sarsa_rewards[-window:])
            sarsa_avg_rewards.append(avg)
            print(f"  Episode {ep+1:6,}: Avg Reward = {avg:6.1f}")
    
    print("\nTraining Monte Carlo...")
    env = RewardShapingWrapper(SnakeEnv(render_mode='text'))
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_data = []
        episode_reward = 0
        
        while not done:
            action = mc_agent.choose_action(state)
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            episode_reward += reward
            state = next_state
        
        mc_agent.update_from_episode(episode_data)
        mc_rewards.append(episode_reward)
        mc_agent.decay_epsilon()
        
        if (ep + 1) % window == 0:
            avg = np.mean(mc_rewards[-window:])
            mc_avg_rewards.append(avg)
            print(f"  Episode {ep+1:6,}: Avg Reward = {avg:6.1f}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nFinal Average Rewards (last {window} episodes):")
    print(f"  Q-Learning:   {q_avg_rewards[-1]:.1f}")
    print(f"  SARSA:        {sarsa_avg_rewards[-1]:.1f}")
    print(f"  Monte Carlo:  {mc_avg_rewards[-1]:.1f}")
    print("="*70 + "\n")
    
    return {
        'episodes': episodes_list,
        'q_learning': q_avg_rewards,
        'sarsa': sarsa_avg_rewards,
        'monte_carlo': mc_avg_rewards
    }


def plot_comparison(data):
    """Create comparison plot of all three algorithms"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    episodes = data['episodes']
    
    # Plot all three algorithms
    ax.plot(episodes, data['q_learning'], linewidth=2.5, color='steelblue', 
            label='Q-Learning', marker='o', markersize=4, markevery=10)
    ax.plot(episodes, data['sarsa'], linewidth=2.5, color='forestgreen', 
            label='SARSA', marker='s', markersize=4, markevery=10)
    ax.plot(episodes, data['monte_carlo'], linewidth=2.5, color='darkorange', 
            label='Monte Carlo', marker='^', markersize=4, markevery=10)
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0)')
    
    # Formatting
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward (500-episode window)', fontsize=14, fontweight='bold')
    ax.set_title('Algorithm Comparison: Learning Curves (20,000 Episodes)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add final performance annotations
    final_q = data['q_learning'][-1]
    final_sarsa = data['sarsa'][-1]
    final_mc = data['monte_carlo'][-1]
    
    stats_text = (
        f"Final Performance:\n"
        f"Q-Learning: {final_q:.1f}\n"
        f"SARSA: {final_sarsa:.1f}\n"
        f"Monte Carlo: {final_mc:.1f}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison plot saved to results/algorithm_comparison.png")
    plt.show()


def plot_final_comparison_bars():
    """Create bar chart comparing final results"""
    
    # Your actual results
    algorithms = ['Q-Learning', 'SARSA', 'Monte Carlo']
    avg_rewards = [100.4, 85.0, 89.5]
    best_rewards = [269.8, 267.8, 258.0]
    avg_lengths = [13.8, 12.1, 12.4]
    best_lengths = [28, 21, 23]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    colors = ['steelblue', 'forestgreen', 'darkorange']
    
    # Average Reward
    bars1 = ax1.bar(x, avg_rewards, width, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Reward', fontweight='bold')
    ax1.set_title('Average Reward (20 Episodes)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(avg_rewards):
        ax1.text(i, v + 2, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Best Reward
    bars2 = ax2.bar(x, best_rewards, width, color=colors, alpha=0.8)
    ax2.set_ylabel('Best Reward', fontweight='bold')
    ax2.set_title('Best Reward Achieved', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(best_rewards):
        ax2.text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Average Length
    bars3 = ax3.bar(x, avg_lengths, width, color=colors, alpha=0.8)
    ax3.set_ylabel('Average Snake Length', fontweight='bold')
    ax3.set_title('Average Snake Length', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Starting Length')
    ax3.legend()
    for i, v in enumerate(avg_lengths):
        ax3.text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Best Length
    bars4 = ax4.bar(x, best_lengths, width, color=colors, alpha=0.8)
    ax4.set_ylabel('Best Snake Length', fontweight='bold')
    ax4.set_title('Maximum Snake Length Achieved', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=59, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Theoretical Max (59)')
    ax4.legend()
    for i, v in enumerate(best_lengths):
        ax4.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Performance comparison saved to results/performance_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE ALGORITHM COMPARISON")
    print("="*70)
    print("\nThis will:")
    print("  1. Train all three algorithms (20k episodes each)")
    print("  2. Generate learning curve comparison plot")
    print("  3. Generate performance comparison charts")
    print("\nEstimated time: 2-3 minutes")
    print("="*70 + "\n")
    
    response = input("Start training? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        exit()
    
    # Train all three and get data
    data = train_all_algorithms(num_episodes=200000, window=500)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_comparison(data)
    plot_final_comparison_bars()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. results/algorithm_comparison.png    (learning curves)")
    print("  2. results/performance_comparison.png  (bar charts)")
    print("="*70 + "\n")