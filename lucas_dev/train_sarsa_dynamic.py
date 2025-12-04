import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from snake_env_dynamic import SnakeEnvDynamic

class SarsaAgentDynamic:
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99995, load_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # State: Food Dir (8), Danger Mask (16), Trap Mask (16)
        # Shape: (8, 16, 16, 4)
        self.q_table = np.zeros((8, 16, 16, action_space_size))
        
        self.total_episodes = 0
        self.reward_history = []
        self.length_history = []
        
        if load_path:
            self.load(load_path)

    def get_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
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
            filename = f"lucas_dev/models/sarsa_dynamic_{self.total_episodes}_episodes.npz"
        np.savez(filename, q_table=self.q_table, total_episodes=self.total_episodes, reward_history=self.reward_history, length_history=self.length_history, epsilon=self.epsilon)
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
            if 'epsilon' in data:
                self.epsilon = float(data['epsilon'])
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: File {filename} not found!")

def plot_training_results(agent, window_size=100):
    if not agent.reward_history: return
    rewards = np.array(agent.reward_history)
    lengths = np.array(agent.length_history) if hasattr(agent, 'length_history') and agent.length_history else np.zeros_like(rewards)

    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        len_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = rewards
        len_avg = lengths

    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    plt.plot(range(len(rewards)-len(moving_avg), len(rewards)), moving_avg, color='red', label='Avg')
    plt.title(f'SARSA Dynamic Training (Episodes: {len(rewards)})')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lengths, alpha=0.3, color='green', label='Raw')
    plt.plot(range(len(lengths)-len(len_avg), len(lengths)), len_avg, color='darkgreen', label='Avg')
    plt.ylabel('Snake Length')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True)

    if not os.path.exists("lucas_dev/plots"): os.makedirs("lucas_dev/plots")
    plt.savefig(f"lucas_dev/plots/training_plot_dynamic_{len(rewards)}.png")
    plt.close()

def train_sarsa_dynamic(agent=None, episodes=20000):
    env = SnakeEnvDynamic(render_mode='text')
    if agent is None: agent = SarsaAgentDynamic()
    
    print(f"Starting Dynamic Reward training for {episodes} episodes...")
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
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        if (episode + 1) % 1000 == 0:
            avg_rew = sum(agent.reward_history[-100:]) / 100
            avg_len = sum(agent.length_history[-100:]) / 100
            print(f"Ep {episode+1}/{episodes} | Total: {agent.total_episodes} | Avg Rew: {avg_rew:.2f} | Avg Len: {avg_len:.2f} | Eps: {agent.epsilon:.4f}")
            
    print(f"Done in {time.time()-start_time:.2f}s")
    agent.save()
    plot_training_results(agent)
    return agent

def visualize_play(agent):
    print("\nVisualizing Dynamic Agent...")
    env = SnakeEnvDynamic(render_mode='pygame', cell_size=60)
    state = env.reset()
    env.render()
    time.sleep(1)
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state, greedy=True)
        next_state, reward, done = env.step(action)
        env.render(delay=100)
        state = next_state
        total_reward += reward
        if env.render_mode == 'pygame':
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT: done = True
    print(f"Score: {total_reward}")
    print(f"Length: {len(env.snake)}")
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    # Train for 200k episodes
    agent = None
    # agent = SarsaAgentDynamic(load_path="lucas_dev/models/sarsa_v3_no_obstacle_200000_episodes.npz")
    agent = train_sarsa_dynamic(agent=agent, episodes=130000)
    visualize_play(agent)
