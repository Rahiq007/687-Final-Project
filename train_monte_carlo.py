"""
Monte Carlo Control for Snake Game
Every-visit MC with epsilon-greedy exploration

Author: Rahiq Majeed
Course: COMPSCI 687 - Fall 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_env_with_vision import SnakeEnv
import time
import pickle
from collections import defaultdict
import os


class RewardShaper:
    """Simple reward shaping"""
    
    def __init__(self, env):
        self.env = env
        self.prev_distance = None
    
    def reset(self):
        state = self.env.reset()
        head_x, head_y = state[0], state[1]
        food_x, food_y = state[2], state[3]
        self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
        return state
    
    def step(self, action):
        next_state, reward, done = self.env.step(action)
        
        head_x, head_y = next_state[0], next_state[1]
        food_x, food_y = next_state[2], next_state[3]
        
        if not done and reward != 10:
            current_distance = abs(head_x - food_x) + abs(head_y - food_y)
            
            if current_distance < self.prev_distance:
                reward += 1.0
            elif current_distance > self.prev_distance:
                reward -= 0.5
            
            self.prev_distance = current_distance
            
        elif reward == 10:
            self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        return next_state, reward, done
    
    def render(self, delay=0):
        return self.env.render(delay)
    
    def close(self):
        self.env.close()
    
    def get_episode_info(self):
        return self.env.get_episode_info()
    
    @property
    def snake(self):
        return self.env.snake


class MonteCarloAgent:
    """Monte Carlo Control - Every-Visit"""
    
    def __init__(self):
        self.gamma = 0.95
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999995
        
        self.Q = defaultdict(lambda: np.zeros(4))
        self.returns = defaultdict(list)  # Store all returns for averaging
        self.visit_counts = defaultdict(int)
    
    def get_q_values(self, state):
        return self.Q[state]
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.get_q_values(state))
    
    def update_from_episode(self, episode):
        """
        Update Q-values from complete episode
        episode: list of (state, action, reward) tuples
        """
        # Calculate returns for each step
        G = 0
        
        # Process episode backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            # Calculate return
            G = reward + self.gamma * G
            
            # Every-visit MC: update for every occurrence
            self.returns[(state, action)].append(G)
            self.Q[state][action] = np.mean(self.returns[(state, action)])
            self.visit_counts[state] += 1
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])
    
    def get_states_count(self):
        return len(self.Q)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'returns': dict(self.returns),
                'epsilon': self.epsilon
            }, f)
        print(f"Agent saved: {filepath}")
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(4), data['Q'])
            self.returns = defaultdict(list, data.get('returns', {}))
            self.epsilon = data.get('epsilon', 0.01)


def train(num_episodes=1000000, print_every=50000):
    """Train Monte Carlo agent"""
    
    print("=" * 70)
    print("MONTE CARLO CONTROL")
    print("=" * 70)
    print(f"Grid: 8x8 | Obstacles: 4 | Max length: 60")
    print(f"State space: 131,072 states")
    print(f"Training: {num_episodes:,} episodes")
    print(f"Algorithm: Every-Visit Monte Carlo")
    print("=" * 70 + "\n")
    
    env = RewardShaper(SnakeEnv(render_mode='text'))
    agent = MonteCarloAgent()
    
    all_rewards = []
    all_lengths = []
    best_length = 0
    best_reward = -999
    
    milestones = [20, 25, 30, 35, 40, 45]
    achieved = set()
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        trajectory = []  # Store (state, action, reward)
        
        # Generate episode
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            trajectory.append((state, action, reward))
            episode_reward += reward
            state = next_state
        
        # Update from complete episode
        agent.update_from_episode(trajectory)
        
        info = env.get_episode_info()
        all_rewards.append(episode_reward)
        all_lengths.append(info['snake_length'])
        
        if info['snake_length'] > best_length:
            best_length = info['snake_length']
            print(f"  🐍 NEW BEST: {best_length} segments! (Episode {episode+1:,})")
            agent.save('best_mc_agent.pkl')
            
            for m in milestones:
                if best_length >= m and m not in achieved:
                    achieved.add(m)
                    print(f"  🎯 MILESTONE: {m} segments!")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        agent.decay_epsilon()
        
        if (episode + 1) % print_every == 0:
            elapsed = time.time() - start_time
            recent_avg_len = np.mean(all_lengths[-print_every:])
            recent_max_len = np.max(all_lengths[-print_every:])
            coverage = agent.get_states_count() / 131072 * 100
            
            eta = elapsed / (episode + 1) * (num_episodes - episode - 1)
            
            print(f"\n{'='*70}")
            print(f"Episode {episode+1:,}/{num_episodes:,} ({(episode+1)/num_episodes*100:.0f}%)")
            print(f"  Recent ({print_every//1000}k episodes):")
            print(f"    Avg Length: {recent_avg_len:.1f}")
            print(f"    Max Length: {recent_max_len}")
            print(f"  All-Time Best:")
            print(f"    Best Length: {best_length} / 60")
            print(f"    Best Reward: {best_reward:.1f}")
            print(f"  Learning:")
            print(f"    Epsilon: {agent.epsilon:.6f}")
            print(f"    States: {agent.get_states_count():,} / 131,072 ({coverage:.1f}%)")
            print(f"  Time: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
    
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best length: {best_length} / 60 ({best_length/60*100:.1f}%)")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Final avg (last 50k): {np.mean(all_lengths[-50000:]):.1f}")
    print(f"States explored: {agent.get_states_count():,} / 131,072 ({agent.get_states_count()/131072*100:.1f}%)")
    print(f"Milestones: {sorted(achieved)}")
    print(f"Total time: {total_time:.1f} min")
    print("=" * 70)
    
    agent.save('final_mc_agent.pkl')
    
    return agent, all_rewards, all_lengths


def evaluate(agent, num_episodes=30, visualize=True):
    """Evaluate trained agent"""
    
    print("\n" + "=" * 70)
    print(f"EVALUATION ({num_episodes} episodes)")
    print("=" * 70)
    
    if visualize:
        env = RewardShaper(SnakeEnv(render_mode='pygame', cell_size=60))
    else:
        env = RewardShaper(SnakeEnv(render_mode='text'))
    
    rewards = []
    lengths = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        
        if visualize:
            env.render(delay=100)
        
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            if visualize:
                env.render(delay=20)
        
        info = env.get_episode_info()
        rewards.append(ep_reward)
        lengths.append(info['snake_length'])
        
        print(f"Episode {ep+1:2d}: Reward={ep_reward:6.1f}, Length={info['snake_length']:3d}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("MONTE CARLO EVALUATION RESULTS")
    print("=" * 70)
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Length: {np.mean(lengths):.1f}")
    print(f"Best Reward: {max(rewards):.1f}")
    print(f"Best Length: {max(lengths)} / 60")
    print(f"Worst Length: {min(lengths)}")
    print(f"Std Dev: {np.std(lengths):.1f}")
    print("=" * 70)
    
    return rewards, lengths


def plot_training(rewards, lengths):
    """Plot training curves"""
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    window = 5000
    
    # Rewards
    ax1 = axes[0, 0]
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, 'b-', linewidth=1)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards (Smoothed)')
    ax1.grid(True, alpha=0.3)
    
    # Lengths
    ax2 = axes[0, 1]
    if len(lengths) >= window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, 'g-', linewidth=1)
    ax2.axhline(60, color='r', linestyle='--', linewidth=2, label='Max (60)')
    ax2.axhline(40, color='orange', linestyle='--', linewidth=1, label='Target (40)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.set_title('Average Snake Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Best length over time
    ax3 = axes[1, 0]
    best = np.maximum.accumulate(lengths)
    ax3.plot(best, 'purple', linewidth=1)
    ax3.axhline(60, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Best Length')
    ax3.set_title('Best Length Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Length distribution (last 10%)
    ax4 = axes[1, 1]
    last = lengths[-len(lengths)//10:]
    ax4.hist(last, bins=40, color='orange', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(last), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(last):.1f}')
    ax4.axvline(40, color='g', linestyle='--', linewidth=1, label='Target: 40')
    ax4.set_xlabel('Length')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Length Distribution (Last 10%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Monte Carlo Training - 8x8 Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/monte_carlo_training_results.png', dpi=150)
    print("Saved: results/monte_carlo_training_results.png")
    plt.show()


def plot_evaluation(rewards, lengths):
    """Plot evaluation results"""
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    episodes = range(1, len(lengths) + 1)
    
    # Lengths
    colors = ['green' if l >= 40 else 'steelblue' for l in lengths]
    ax1.bar(episodes, lengths, color=colors, alpha=0.7)
    ax1.axhline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Avg: {np.mean(lengths):.1f}')
    ax1.axhline(60, color='green', linestyle='--', linewidth=2, label='Max: 60')
    ax1.axhline(40, color='orange', linestyle='--', linewidth=1, label='Target: 40')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Length')
    ax1.set_title('Evaluation - Snake Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Rewards
    ax2.bar(episodes, rewards, color='orange', alpha=0.7)
    ax2.axhline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Avg: {np.mean(rewards):.1f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Evaluation - Total Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Monte Carlo Evaluation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/monte_carlo_evaluation_results.png', dpi=150)
    print("Saved: results/monte_carlo_evaluation_results.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 70)
    print("MONTE CARLO ALGORITHM")
    print("Training: 1,000,000 episodes (~20-25 minutes)")
    print("=" * 70 + "\n")
    
    # Train
    agent, rewards, lengths = train(num_episodes=1000000, print_every=50000)
    
    # Plot training
    plot_training(rewards, lengths)
    
    # Evaluate
    eval_rewards, eval_lengths = evaluate(agent, num_episodes=30, visualize=True)
    
    # Plot evaluation
    plot_evaluation(eval_rewards, eval_lengths)
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)