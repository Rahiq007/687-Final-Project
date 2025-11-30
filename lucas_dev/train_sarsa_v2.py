import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from snake_env_v2 import SnakeEnvV2

class SarsaAgentV2:
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.99, epsilon=0.1, load_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # State: head_x, head_y, food_x, food_y, danger_mask
        # Grid size is 8x8
        # Shape: (8, 8, 8, 8, 16, action_space_size)
        self.q_table = np.zeros((8, 8, 8, 8, 16, action_space_size))
        
        self.total_episodes = 0
        self.reward_history = []
        
        if load_path:
            self.load(load_path)

    def get_action(self, state):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def save(self, filename=None):
        if not os.path.exists("lucas_dev/models"):
            os.makedirs("lucas_dev/models")
            
        if filename is None:
            filename = f"lucas_dev/models/sarsa_v2_{self.total_episodes}_episodes.npz"
            
        np.savez(filename, q_table=self.q_table, total_episodes=self.total_episodes, reward_history=self.reward_history)
        print(f"Model saved to {filename}")
        return filename

    def load(self, filename):
        if os.path.exists(filename):
            data = np.load(filename)
            self.q_table = data['q_table']
            self.total_episodes = int(data['total_episodes'])
            if 'reward_history' in data:
                self.reward_history = list(data['reward_history'])
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: File {filename} not found!")

def plot_training_results(agent, window_size=100):
    """Plot the moving average of rewards"""
    if not agent.reward_history:
        print("No reward history to plot.")
        return
        
    rewards = np.array(agent.reward_history)
    
    # Calculate moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = rewards
        
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    plt.plot(range(len(rewards)-len(moving_avg), len(rewards)), moving_avg, color='red', label=f'{window_size}-Episode Moving Avg')
    
    plt.title(f'SARSA Training Progress (Total Episodes: {len(rewards)})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    if not os.path.exists("lucas_dev/plots"):
        os.makedirs("lucas_dev/plots")
    
    plot_path = f"lucas_dev/plots/training_plot_{len(rewards)}.png"
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()

def train_sarsa_v2(agent=None, episodes=20000):
    env = SnakeEnvV2(render_mode='text')
    
    if agent is None:
        agent = SarsaAgentV2()
        print(f"Starting new V2 training session for {episodes} episodes...")
    else:
        print(f"Continuing V2 training for {episodes} more episodes...")
    
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
            
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(agent.reward_history[-100:]) / 100
            print(f"Episode {episode + 1}/{episodes} | Total: {agent.total_episodes} | Avg Reward (last 100): {avg_reward:.2f}")
    
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds")
    
    # Save model and plot
    agent.save()
    plot_training_results(agent)
            
    return agent

def visualize_play(agent):
    print("\nVisualizing agent play...")
    env = SnakeEnvV2(render_mode='pygame', cell_size=60)
        
    state = env.reset()
    env.render()
    time.sleep(1)
    
    done = False
    total_reward = 0
    
    while not done:
        # Greedy action
        action = np.argmax(agent.q_table[state])
        next_state, reward, done = env.step(action)
        
        env.render(delay=100)
        
        state = next_state
        total_reward += reward
        
        if env.render_mode == 'pygame':
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
    
    print(f"Game Over! Score: {total_reward}")
    print(f"Final Snake Length: {len(env.snake)}")
    time.sleep(15)
    env.close()

if __name__ == "__main__":
    # Train
    agent = train_sarsa_v2(episodes=300000)
    
    # Visualize
    visualize_play(agent)
