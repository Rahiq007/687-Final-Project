import numpy as np
import random
import time
import os
from snake_env import SnakeEnv

class SarsaAgent:
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # State: head_x, head_y, food_x, food_y, direction
        # Grid size is 8x8 as defined in SnakeEnv
        # Shape: (8, 8, 8, 8, 4, action_space_size)
        self.q_table = np.zeros((8, 8, 8, 8, 4, action_space_size))
        
        # Track training progress
        self.total_episodes = 0

    def get_action(self, state):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        # SARSA update rule: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def save(self, filename=None):
        """
        Save the Q-table and metadata to a file.
        If no filename is provided, saves to 'models/sarsa_snake_{episodes}_episodes.npz'
        """
        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
            
        if filename is None:
            filename = f"models/sarsa_snake_{self.total_episodes}_episodes.npz"
            
        # Save both the Q-table and the episode count
        np.savez(filename, q_table=self.q_table, total_episodes=self.total_episodes)
        print(f"Model saved to {filename}")
        return filename

    def load(self, filename):
        """Load the Q-table and metadata from a file"""
        if os.path.exists(filename):
            try:
                data = np.load(filename)
                # Check if it's the new .npz format or old .npy format
                if filename.endswith('.npz'):
                    self.q_table = data['q_table']
                    self.total_episodes = int(data['total_episodes'])
                    print(f"Model loaded from {filename} (Previously trained for {self.total_episodes} episodes)")
                else:
                    # Fallback for old .npy files (just the array)
                    self.q_table = data
                    # Try to guess episodes from filename if possible, else 0
                    try:
                        # expected format: sarsa_snake_{number}_steps.npy
                        parts = filename.split('_')
                        for part in parts:
                            if part.isdigit():
                                self.total_episodes = int(part)
                                break
                    except:
                        pass
                    print(f"Legacy model loaded from {filename}. Assumed episodes: {self.total_episodes}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Error: File {filename} not found!")

def train_sarsa(agent=None, episodes=20000):
    """
    Train the SARSA agent.
    
    Args:
        agent: Existing SarsaAgent to continue training (optional)
        episodes: Number of episodes to train for
    """
    # Use text mode for faster training (no graphics)
    env = SnakeEnv(render_mode='text')
    
    if agent is None:
        agent = SarsaAgent()
        print(f"Starting new training session for {episodes} episodes...")
    else:
        print(f"Continuing training for {episodes} more episodes (Current: {agent.total_episodes})...")
    
    start_episodes = agent.total_episodes
    
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
            
        if (episode + 1) % 1000 == 0:
            print(f"Session Episode {episode + 1}/{episodes} | Total Episodes: {agent.total_episodes}")
    
    # Save the agent automatically
    saved_file = agent.save()
            
    return agent, saved_file

def visualize_play(agent):
    print("\nVisualizing agent play...")
    # Use pygame for visualization if available
    env = SnakeEnv(render_mode='pygame', cell_size=60)
        
    state = env.reset()
    env.render()
    time.sleep(1)
    
    done = False
    total_reward = 0
    
    while not done:
        # Use greedy action for demonstration (exploitation only)
        action = np.argmax(agent.q_table[state])
        next_state, reward, done = env.step(action)
        
        # Add delay to make it visible to human eye
        env.render(delay=100)
        
        state = next_state
        total_reward += reward
        
        # Allow closing the window
        if env.render_mode == 'pygame':
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
    
    print(f"Game Over! Score: {total_reward}")
    time.sleep(1)
    env.close()

if __name__ == "__main__":
    print("="*60)
    print("SARSA TRAINING WITH SAVE/LOAD")
    print("="*60)
    
    file1 = r"models/sarsa_snake_460000_episodes.npz"
    # Create a fresh agent instance
    loaded_agent = SarsaAgent()
    # Load the previous state
    loaded_agent.load(file1)
    
    # # Train for another 20,000 episodes
    print(f"Training for another 20,000 episodes...")
    loaded_agent, file2 = train_sarsa(agent=loaded_agent, episodes=200000)
        
    # 3. Watch it play
    visualize_play(loaded_agent)