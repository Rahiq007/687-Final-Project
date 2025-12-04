import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from snake_env_no_obstacle import SnakeEnvNoObstacle

# Check for MPS (Apple Silicon) or CUDA
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, action_size=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

class CNNAgent:
    def __init__(self, action_size=4, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, load_path=None):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.model = SimpleCNN(input_channels=3, action_size=action_size).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.total_episodes = 0
        self.reward_history = []
        self.length_history = []  # Track snake length
        self.step_history = []    # Track steps per episode
        
        if load_path:
            self.load(load_path)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, next_action, done):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        reward_t = torch.FloatTensor([reward]).to(device)
        
        current_q = self.model(state_t)[0, action].unsqueeze(0)
        
        with torch.no_grad():
            if done:
                target_q = reward_t
            else:
                next_q = self.model(next_state_t)[0, next_action]
                target_q = reward_t + self.gamma * next_q
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, filename=None):
        if not os.path.exists("lucas_dev/models"):
            os.makedirs("lucas_dev/models")
        if filename is None:
            filename = f"lucas_dev/models/cnn_no_obstacle_{self.total_episodes}_episodes.pth"
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'reward_history': self.reward_history,
            'length_history': self.length_history,
            'step_history': self.step_history,
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_episodes = checkpoint['total_episodes']
            self.reward_history = checkpoint.get('reward_history', [])
            self.length_history = checkpoint.get('length_history', [])
            self.step_history = checkpoint.get('step_history', [])
            self.epsilon = checkpoint.get('epsilon', 0.01)
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: File {filename} not found!")

def plot_training_results(agent, window_size=100):
    if not agent.reward_history: return
    
    # Plot Rewards
    rewards = np.array(agent.reward_history)
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = rewards
        
    plt.figure(figsize=(10, 15))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    plt.plot(range(len(rewards)-len(moving_avg), len(rewards)), moving_avg, color='red', label='Avg')
    plt.title(f'CNN No Obstacle Training (Episodes: {len(rewards)})')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot Lengths
    lengths = np.array(agent.length_history)
    if len(lengths) >= window_size:
        len_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
    else:
        len_avg = lengths
        
    plt.subplot(3, 1, 2)
    plt.plot(lengths, alpha=0.3, color='green', label='Raw')
    plt.plot(range(len(lengths)-len(len_avg), len(lengths)), len_avg, color='darkgreen', label='Avg')
    plt.ylabel('Snake Length')
    plt.legend()
    plt.grid(True)

    # Plot Steps
    steps = np.array(agent.step_history)
    if len(steps) >= window_size:
        step_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
    else:
        step_avg = steps
        
    plt.subplot(3, 1, 3)
    plt.plot(steps, alpha=0.3, color='purple', label='Raw')
    plt.plot(range(len(steps)-len(step_avg), len(steps)), step_avg, color='indigo', label='Avg')
    plt.ylabel('Steps per Episode')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists("lucas_dev/plots"): os.makedirs("lucas_dev/plots")
    plt.savefig(f"lucas_dev/plots/training_plot_cnn_no_obstacle_neg_{len(rewards)}.png")
    plt.close()

def train_cnn_parallel(agent=None, episodes=20000):
    env = SnakeEnvNoObstacle(render_mode='text')
    if agent is None: agent = CNNAgent()
    
    print(f"Starting CNN training (No Obstacles) for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        
        # Epsilon-greedy
        if random.random() < agent.epsilon:
            action = random.randint(0, 3)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = agent.model(state_t)
            action = torch.argmax(q).item()
            
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            
            # Next action (SARSA)
            if random.random() < agent.epsilon:
                next_action = random.randint(0, 3)
            else:
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    next_q = agent.model(next_state_t)
                next_action = torch.argmax(next_q).item()
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            
        agent.total_episodes += 1
        agent.reward_history.append(env.total_reward)
        agent.length_history.append(len(env.snake)) # Track Length
        agent.step_history.append(env.steps)        # Track Steps
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        if (episode + 1) % 100 == 0:
            avg_rew = sum(agent.reward_history[-100:]) / 100
            avg_len = sum(agent.length_history[-100:]) / 100
            avg_steps = sum(agent.step_history[-100:]) / 100
            print(f"Ep {episode+1}/{episodes} | Avg Rew: {avg_rew:.2f} | Avg Len: {avg_len:.2f} | Avg Steps: {avg_steps:.2f} | Eps: {agent.epsilon:.3f}")
            
    print(f"Done in {time.time()-start_time:.2f}s")
    agent.save()
    plot_training_results(agent)
    return agent

def visualize_play(agent):
    print("\nVisualizing CNN Agent (No Obstacles)...")
    env = SnakeEnvNoObstacle(render_mode='pygame', cell_size=60)
    state = env.reset()
    env.render()
    time.sleep(1)
    done = False
    total_reward = 0
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    while not done:
        action = agent.get_action(state)
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
    agent.epsilon = original_epsilon
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    # Set start method for multiprocessing (required for PyTorch on Mac)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    # Train
    agent = None
    agent = CNNAgent(load_path="/Users/xiaofanlu/Documents/github_repos/687-Final-Project/lucas_dev/models/cnn_no_obstacle_9000_episodes.pth")
    # Since we changed the state representation (negative body), we should train from scratch
    # agent = train_cnn_parallel(agent=agent, episodes=7100) 
    visualize_play(agent)
