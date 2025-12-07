
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
from snake_env_v7 import SnakeEnvV7

class SarsaAgentV7:
    """
    SARSA Agent for SnakeEnvV7
    State Shape: (8, 256) -> (Food, Surround)
    Total States: 2048
    """
    def __init__(self, action_space_size=4, alpha=0.1, gamma=0.99, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.99995, load_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # State: Food(8), Surround(256)
        # Action: 4
        self.q_table = np.zeros((8, 256, action_space_size))
        
        self.total_episodes = 0
        self.reward_history = []
        self.length_history = []
        
        if load_path:
            self.load(load_path)

    def get_action(self, state, forbidden_action=None):
        if random.random() < self.epsilon:
            valid_actions = [0, 1, 2, 3]
            if forbidden_action is not None and forbidden_action in valid_actions:
                valid_actions.remove(forbidden_action)
            return random.choice(valid_actions)
        
        return self.get_greedy_action(state, forbidden_action)

    def get_greedy_action(self, state, forbidden_action=None):
        q_values = self.q_table[state].copy()
        if forbidden_action is not None:
             q_values[forbidden_action] = -np.inf
        return np.argmax(q_values)

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
            filename = f"lucas_dev/models/sarsa_v7_{self.total_episodes}_episodes.npz"
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
    plt.title(f'SARSA V7 Training (Episodes: {len(rewards)})')
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
    plt.savefig(f"lucas_dev/plots/training_plot_v7_{len(rewards)}.png")
    plt.close()

def plot_evaluation_results(rewards, lengths, steps):
    """Create single clean results graph for evaluation (Length focused)"""
    if plt is None: return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    episodes = range(1, len(lengths) + 1)
    
    # Plot length as bars
    bars = ax.bar(episodes, lengths, color='mediumseagreen', alpha=0.7, label='Snake Length')
    
    # Add average line
    avg_length = np.mean(lengths)
    ax.axhline(y=avg_length, color='darkgreen', linestyle='--', linewidth=2,
               label=f'Average: {avg_length:.1f}')
    
    # Formatting
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Snake Length', fontsize=13, fontweight='bold')
    ax.set_title('SARSA V7 Evaluation Results - Snake Length', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    stats_text = (
        f"Statistics:\n"
        f"Avg Length: {np.mean(lengths):.1f}\n"
        f"Best Length: {max(lengths)}\n"
        f"Avg Reward: {np.mean(rewards):.1f}\n"
        f"Best Reward: {max(rewards):.1f}\n"
        f"Avg Steps: {np.mean(steps):.1f}"
    )
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    if not os.path.exists("lucas_dev/plots"):
        os.makedirs("lucas_dev/plots")
    
    plt.tight_layout()
    plt.savefig('lucas_dev/plots/sarsa_v7_evaluation_length.png', dpi=150, bbox_inches='tight')
    print("✓ Results saved to lucas_dev/plots/sarsa_v7_evaluation_length.png\n")
    plt.close()

def get_forbidden_action(direction):
    """Returns the opposite direction which is forbidden (suicide)."""
    if direction == 0: return 1 # UP -> DOWN
    if direction == 1: return 0 # DOWN -> UP
    if direction == 2: return 3 # LEFT -> RIGHT
    if direction == 3: return 2 # RIGHT -> LEFT
    return None

def evaluate_agent(agent, num_episodes=20):
    """Evaluate trained agent and create results visualization"""
    
    print("="*70)
    print("EVALUATING TRAINED AGENT (V7)")
    print("="*70)
    
    # Use SnakeEnvV7
    env = SnakeEnvV7(render_mode='pygame', cell_size=60)
    
    rewards = []
    lengths = []
    steps_list = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Render initial state
        env.render(delay=500)
        
        while not done:
            forbidden = get_forbidden_action(env.direction)
            action = agent.get_greedy_action(state, forbidden_action=forbidden)
            
            # Helper to interpret actions/directions
            dir_names = ["UP", "DOWN", "LEFT", "RIGHT"]
            last_dir = env.direction
            last_head = env.snake[0]
            
            state, reward, done = env.step(action)
            episode_reward += reward
            env.render(delay=50) # Faster playback for evaluation
            
            # Handle pygame events to prevent freezing
            if env.render_mode == 'pygame':
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            
            if done:
                # Calculate where the snake tried to go
                hx, hy = last_head
                if action == 0: target_hit = (hx, hy - 1)
                elif action == 1: target_hit = (hx, hy + 1)
                elif action == 2: target_hit = (hx - 1, hy)
                else: target_hit = (hx + 1, hy)
                
                # Determine cause of death
                cause = "UNKNOWN"
                tx, ty = target_hit
                if tx < 0 or tx >= env.grid_size or ty < 0 or ty >= env.grid_size:
                    cause = "WALL"
                elif target_hit in env.obstacles:
                    cause = "OBSTACLE"
                elif target_hit in env.snake:
                    cause = "SELF_COLLISION"

                print(f"  [Death Debug] Cause: {cause} | "
                      f"Prev Dir: {dir_names[last_dir] if last_dir is not None else 'None'} | "
                      f"Action: {dir_names[action]} | "
                      f"Head: {last_head} -> Hit: {target_hit}")

                print("   Final Grid State:")
                grid_str = "   " + " ".join([str(i) for i in range(env.grid_size)]) + "\n"
                for y in range(env.grid_size):
                    grid_str += f"{y}  "
                    for x in range(env.grid_size):
                        p = (x, y)
                        if p == env.food:
                            char = "F" # Food
                        elif p == env.snake[0]:
                            char = "H" # Head
                        elif p in env.snake:
                            char = "B" # Body
                        elif p in env.obstacles:
                            char = "X" # Obstacle
                        else:
                            char = "." # Empty
                        grid_str += f"{char} "
                    grid_str += "\n"
                print(grid_str)
        
        # Collect stats
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
    print(f"Best Reward: {max(rewards):.1f}")
    print(f"Best Length: {max(lengths)}")
    print("="*70 + "\n")
    
    # Create results graph
    plot_evaluation_results(rewards, lengths, steps_list)
    
    return rewards, lengths, steps_list

def train_sarsa_v7(agent=None, episodes=20000):
    env = SnakeEnvV7(render_mode='text')
    if agent is None: agent = SarsaAgentV7()
    
    print(f"Starting V7 training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        forbidden = get_forbidden_action(env.direction)
        action = agent.get_action(state, forbidden_action=forbidden)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            
            next_forbidden = get_forbidden_action(env.direction)
            next_action = agent.get_action(next_state, forbidden_action=next_forbidden)
            
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            
        agent.total_episodes += 1
        agent.reward_history.append(env.total_reward)
        agent.length_history.append(len(env.snake))
        agent.decay_epsilon()
        
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(agent.reward_history[-100:]) / 100
            avg_length = sum(agent.length_history[-100:]) / 100
            print(f"Ep {episode+1}/{episodes} | Total: {agent.total_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Epsilon: {agent.epsilon:.4f}")
            
    print(f"Done in {time.time()-start_time:.2f}s")
    agent.save()
    plot_training_results(agent)
    return agent

if __name__ == "__main__":
    # Train for 50k episodes (5000 states is small enough for 50k)
    agent = SarsaAgentV7(load_path="lucas_dev/models/sarsa_v7_100000_episodes.npz")
    # agent = train_sarsa_v7(episodes=100000)
    evaluate_agent(agent, num_episodes=10)
