import sys
import os
import pickle
# Add parent directory to path to allow importing visualize
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from snake_env_v4 import SnakeEnvV4

class SarsaAgentV4:
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.99, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.99995, load_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # State: Head(64), Food(64), HeadDir(4), TailDir(8), Hazard(16)
        # Action: Absolute(4)
        # Shape: (64, 64, 4, 8, 16, 4)
        # Size: ~2M * 4 * 8 bytes ≈ 64MB (Fits in memory easily)
        self.q_table = np.zeros((64, 64, 4, 8, 16, action_space_size))
        
        self.total_episodes = 0
        self.reward_history = []
        self.length_history = []
        
        if load_path:
            self.load(load_path)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def get_greedy_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (target - current_q)
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename=None):
        if not os.path.exists("lucas_dev/models"):
            os.makedirs("lucas_dev/models")
        if filename is None:
            filename = f"lucas_dev/models/sarsa_v4_{self.total_episodes}_episodes.npz"
        
        # Save as compressed numpy
        np.savez(filename, 
                 q_table=self.q_table, 
                 total_episodes=self.total_episodes, 
                 reward_history=self.reward_history, 
                 length_history=self.length_history)
        print(f"Model saved to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            data = np.load(filename)
            self.q_table = data['q_table']
            self.total_episodes = int(data['total_episodes'])
            if 'reward_history' in data:
                self.reward_history = list(data['reward_history'])
            if 'length_history' in data:
                self.length_history = list(data['length_history'])
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: File {filename} not found!")

def plot_training_results(agent, window_size=100):
    if not agent.reward_history: return
    rewards = np.array(agent.reward_history)
    lengths = np.array(agent.length_history) if hasattr(agent, 'length_history') and agent.length_history else np.zeros_like(rewards)

    if len(rewards) >= window_size:
        moving_avg_reward = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_length = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg_reward = rewards
        moving_avg_length = lengths

    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    plt.plot(range(len(rewards)-len(moving_avg_reward), len(rewards)), moving_avg_reward, color='red', label='Avg Reward')
    plt.title(f'SARSA V4 (Head/Food Pos + Hazard) Training')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lengths, alpha=0.3, color='green', label='Raw Length')
    plt.plot(range(len(lengths)-len(moving_avg_length), len(lengths)), moving_avg_length, color='darkgreen', label='Avg Length')
    plt.ylabel('Snake Length')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True)

    if not os.path.exists("lucas_dev/plots"): os.makedirs("lucas_dev/plots")
    plt.savefig(f"lucas_dev/plots/training_plot_v4_pos_{len(rewards)}.png")
    plt.close()

def evaluate_agent(agent, num_episodes=20):
    print("="*70)
    print("EVALUATING TRAINED AGENT (V4 Positional)")
    print("="*70)
    
    env = SnakeEnvV4(render_mode='pygame', cell_size=60)
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
            env.render(delay=50) # Faster replay
            
            if env.render_mode == 'pygame':
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: done = True
        
        current_length = len(env.snake)
        current_steps = env.steps
        rewards.append(episode_reward)
        lengths.append(current_length)
        steps_list.append(current_steps)
        print(f"Episode {ep+1:2d}: Reward={episode_reward:7.1f}, Length={current_length:2d}, Steps={current_steps:3d}")
    
    env.close()
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print("="*70 + "\n")
    return rewards, lengths, steps_list

def train_sarsa_v4(agent=None, episodes=400000): # Increased episodes for 2M states
    env = SnakeEnvV4(render_mode='text')
    if agent is None: agent = SarsaAgentV4()
    
    # Needs significant exploration for 2M states
    agent.epsilon = 0.6
    agent.epsilon_decay = 0.99998 # Very slow decay
    
    print(f"Starting V4 (Positional) training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.get_action(state)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            
        agent.total_episodes += 1
        agent.reward_history.append(env.total_reward)
        agent.length_history.append(len(env.snake))
        agent.decay_epsilon()
        
        if (episode + 1) % 5000 == 0:
            avg_reward = sum(agent.reward_history[-100:]) / 100
            avg_length = sum(agent.length_history[-100:]) / 100
            print(f"Ep {episode+1}/{episodes} | Avg Rew: {avg_reward:.2f} | Avg Len: {avg_length:.2f} | Eps: {agent.epsilon:.4f}")
            
    print(f"Done in {time.time()-start_time:.2f}s")
    agent.save()
    plot_training_results(agent)
    return agent

if __name__ == "__main__":
    # Train for 400k (2M states needs time)
    # agent = SarsaAgentV4(load_path="lucas_dev/models/sarsa_v4_250000_episodes.npz")
    agent = train_sarsa_v4(episodes=400000)
    evaluate_agent(agent, num_episodes=10)
