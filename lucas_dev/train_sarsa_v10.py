
import sys
import os
# Add parent directory to path to allow importing visualize
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import time
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from snake_env_v10 import SnakeEnvV10

class SarsaAgentV10:
    """
    SARSA Agent for SnakeEnvV10
    State Shape: (8, 256, 3) -> (Food, Surround, LengthCat)
    Total States: 6144
    """
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.99, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.99995, load_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # State: Food(8), Surround(256), Length(3)
        self.q_table = np.zeros((8, 256, 3, action_space_size))
        
        self.total_episodes = 0
        self.reward_history = []
        self.length_history = []
        
        if load_path:
            self.load(load_path)

    def get_action(self, state):
        # Standard Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Greedy selection
        return self.get_greedy_action(state)

    def get_greedy_action(self, state, invalid_actions=None):
        q_values = self.q_table[state].copy()
        
        # Resolve ties randomly
        max_val = np.max(q_values)
        best_actions = [i for i, v in enumerate(q_values) if v == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (target - current_q)
        
    def save(self, filename=None):
        if not os.path.exists("lucas_dev/models"):
            os.makedirs("lucas_dev/models")
        if filename is None:
            filename = f"lucas_dev/models/sarsa_v10_{self.total_episodes}_episodes.npz"
        np.savez(filename, q_table=self.q_table, total_episodes=self.total_episodes, reward_history=self.reward_history, length_history=self.length_history)
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
    if plt is None: return
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
    plt.plot(rewards, alpha=0.3, color='steelblue', label='Raw Reward')
    plt.plot(range(len(rewards)-len(moving_avg_reward), len(rewards)), moving_avg_reward, color='darkblue', label='Avg Reward')
    plt.title(f'SARSA V10 Training (Episodes: {len(rewards)})')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lengths, alpha=0.3, color='lightgreen', label='Raw Length')
    plt.plot(range(len(lengths)-len(moving_avg_length), len(lengths)), moving_avg_length, color='darkgreen', label='Avg Length')
    plt.ylabel('Snake Length')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True)

    if not os.path.exists("lucas_dev/plots"): os.makedirs("lucas_dev/plots")
    plt.savefig(f"lucas_dev/plots/training_plot_v10_{len(rewards)}.png")
    plt.close()

def plot_evaluation_results(rewards, lengths, steps):
    if plt is None: return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    episodes = range(1, len(lengths) + 1)
    bars = ax.bar(episodes, lengths, color='mediumseagreen', alpha=0.7, label='Snake Length')
    avg_length = np.mean(lengths)
    ax.axhline(y=avg_length, color='darkgreen', linestyle='--', linewidth=2,
               label=f'Average: {avg_length:.1f}')
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Snake Length', fontsize=13, fontweight='bold')
    ax.set_title('SARSA V10 Evaluation Results - Snake Length', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    stats_text = (
        f"Statistics:\n"
        f"Avg Length: {np.mean(lengths):.1f}\n"
        f"Best Length: {max(lengths)}\n"
        f"Avg Reward: {np.mean(rewards):.1f}\n"
        f"Best Reward: {max(rewards):.1f}\n"
        f"Avg Steps: {np.mean(steps):.1f}"
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    if not os.path.exists("lucas_dev/plots"): os.makedirs("lucas_dev/plots")
    plt.tight_layout()
    plt.savefig('lucas_dev/plots/sarsa_v10_evaluation_length.png', dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_agent(agent, num_episodes=10):
    print("="*70)
    print("EVALUATING TRAINED AGENT (V10)")
    print("="*70)
    
    env = SnakeEnvV10(render_mode='pygame', cell_size=60)
    rewards = []
    lengths = []
    steps_list = []
    
    try:
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            print(f"\n\nEpisode {ep+1}:")
            
            while not done:
                action = agent.get_action(state)
                
                # Helper to interpret actions/directions
                dir_names = ["UP", "DOWN", "LEFT", "RIGHT"]
                last_dir = env.direction
                last_head = env.snake[0]

                # Capture state BEFORE move
                pre_grid_str = "   " + " ".join([str(i) for i in range(env.grid_size)]) + "\n"
                for y in range(env.grid_size):
                    pre_grid_str += f"{y}  "
                    for x in range(env.grid_size):
                        p = (x, y)
                        if p == env.food:
                            char = "F"
                        elif p == env.snake[0]:
                            char = "H"
                        elif p in env.snake:
                            char = "B"
                        elif p in env.obstacles:
                            char = "X"
                        else:
                            char = "."
                        pre_grid_str += f"{char} "
                    pre_grid_str += "\n"
                
                state, reward, done = env.step(action)
                episode_reward += reward
                env.render(delay=50) 
                
                if env.render_mode == 'pygame':
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            return 
                
                if done:
                    # Calculate where the snake tried to go
                    hx, hy = last_head
                    target_hit = None
                    if action == 0: target_hit = (hx, hy - 1)
                    elif action == 1: target_hit = (hx, hy + 1)
                    elif action == 2: target_hit = (hx - 1, hy)
                    else: target_hit = (hx + 1, hy)
                    
                    # Determine cause of death
                    cause = "UNKNOWN"
                    tx, ty = target_hit
                    if env.current_hunger <= 0:
                        cause = "HUNGER/STARVATION"
                    elif env.steps >= env.MAX_STEPS:
                        cause = "TIMEOUT"
                    elif tx < 0 or tx >= env.grid_size or ty < 0 or ty >= env.grid_size:
                        cause = "WALL"
                    elif target_hit in env.obstacles:
                        cause = "OBSTACLE"
                    elif target_hit in env.snake:
                        cause = "SELF_COLLISION"
                    elif env.food is None:
                        cause = "VICTORY"

                    print(f"  [Death Debug] Cause: {cause} | "
                          f"Prev Dir: {dir_names[last_dir] if last_dir is not None else 'None'} | "
                          f"Action: {dir_names[action]} | "
                          f"Head: {last_head} -> Hit: {target_hit}")
                    
                    print(f"   State info: FoodDir={state[0]}, Surround={state[1]}, LenCat={state[2]}")
                    
                    print("   State Before Fatal Move:")
                    print(pre_grid_str)
            
            rewards.append(episode_reward)
            lengths.append(len(env.snake))
            steps_list.append(env.steps)
            print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Length={len(env.snake)}")

        agent.epsilon = old_epsilon

    finally:
        env.close()
    
    if lengths:
        print(f"Avg Length: {np.mean(lengths)}")
        plot_evaluation_results(rewards, lengths, steps_list)

def train_sarsa_v10(agent=None, episodes=40000):
    env = SnakeEnvV10(render_mode='text')
    if agent is None: agent = SarsaAgentV10()
    
    print(f"Starting V10 training for {episodes} episodes...")
    start_time = time.time()
    
    # Custom decay: 70% explore, linear drop from 1.0 to epsilon_min
    start_epsilon = 1.0
    exploration_cutoff = int(episodes * 0.7)
    
    for episode in range(episodes):
        # Calculate dynamic epsilon
        if episode < exploration_cutoff:
             progress = episode / exploration_cutoff
             agent.epsilon = start_epsilon - (progress * (start_epsilon - agent.epsilon_min))
        else:
             agent.epsilon = agent.epsilon_min

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
        
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(agent.reward_history[-100:]) / 100
            avg_length = sum(agent.length_history[-100:]) / 100
            print(f"Ep {episode+1}/{episodes} | Total: {agent.total_episodes} | Avg Rew: {avg_reward:.2f} | Avg Len: {avg_length:.2f} | Eps: {agent.epsilon:.4f}")
            
    print(f"Done in {time.time()-start_time:.2f}s")
    agent.save()
    plot_training_results(agent)
    return agent

if __name__ == "__main__":
    agent = train_sarsa_v10(episodes=50000)
    evaluate_agent(agent, num_episodes=10)
