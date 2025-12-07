
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
from snake_env_v8 import SnakeEnvV8

class SarsaAgentV8:
    """
    SARSA Agent for SnakeEnvV8 (Safety First)
    State Shape: (8, 256) -> (Food, Surround)
    Total States: 2048
    
    Key Improvement:
    - Explicitly decodes the state to identify 'Immediate Death' (0) actions.
    - Masks these actions during selection (both Epsilon-Greedy and Greedy).
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

    def get_safety_status(self, state):
        """
        Decodes the surround state (base-4) and returns a dict mapping actions to safety levels.
        Levels:
        0: Immediate Death
        1: Trap
        2: Safe (Neutral)
        3: Safe (Closer)
        """
        _, surround_code = state
        status = {}
        for i in range(4):
            val = (surround_code // (4**i)) % 4
            status[i] = val
        return status

    def get_action(self, state, forbidden_action=None):
        safety = self.get_safety_status(state)
        
        # Define sets of actions based on safety
        safe_actions = [a for a, s in safety.items() if s >= 2]
        trap_actions = [a for a, s in safety.items() if s == 1]
        death_actions = [a for a, s in safety.items() if s == 0]
        
        # Filter forbidden (reverse) from the best available tier
        # However, never filter if it leaves us with nothing or forces a drop in tier
        # Actually simplest is: Filter forbidden from all sets, then pick best set.
        
        candidates = []
        if forbidden_action is not None:
            # Try to exclude forbidden from safe
            filtered_safe = [a for a in safe_actions if a != forbidden_action]
            if filtered_safe:
                candidates = filtered_safe
            elif safe_actions:
                candidates = safe_actions # Must pick forbidden (but safe)
            else:
                # No safe actions. Try Traps.
                filtered_traps = [a for a in trap_actions if a != forbidden_action]
                if filtered_traps:
                    candidates = filtered_traps
                elif trap_actions:
                    candidates = trap_actions
                else:
                    candidates = death_actions # Doom
        else:
            if safe_actions: candidates = safe_actions
            elif trap_actions: candidates = trap_actions
            else: candidates = death_actions
            
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.choice(candidates)
        
        # Greedy selection restricted to candidates
        # We need to construct the list of invalid actions (everything NOT in candidates) for the greedy helper
        all_actions = [0, 1, 2, 3]
        invalid_actions = [a for a in all_actions if a not in candidates]
        
        return self.get_greedy_action(state, invalid_actions)

    def get_greedy_action(self, state, invalid_actions=None):
        if invalid_actions is None:
            invalid_actions = []
        
        q_values = self.q_table[state].copy()
        
        # Mask invalids
        for action in invalid_actions:
             q_values[action] = -np.inf
             
        # Check if all masked (should be handled by caller properly, but safety fallback)
        if np.all(q_values == -np.inf):
             # Restore originals? Or just picking 0 is fine since we are doomed.
             # Let's restore to pick the "least bad" according to Q-table even if masked
             q_values = self.q_table[state].copy()
        
        # Argmax breaks ties by picking first index. 
        # To avoid bias, we can shuffle? Q-learning usually deterministic greedy.
        # But dealing with uninitialized zones (0s):
        max_val = np.max(q_values)
        best_actions = [i for i, v in enumerate(q_values) if v == max_val]
        return random.choice(best_actions)

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
            filename = f"lucas_dev/models/sarsa_v8_{self.total_episodes}_episodes.npz"
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
    plt.title(f'SARSA V8 Training (Episodes: {len(rewards)})')
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
    plt.savefig(f"lucas_dev/plots/training_plot_v8_{len(rewards)}.png")
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
    ax.set_title('SARSA V8 Evaluation Results - Snake Length', 
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
    plt.savefig('lucas_dev/plots/sarsa_v8_evaluation_length.png', dpi=150, bbox_inches='tight')
    print("✓ Results saved to lucas_dev/plots/sarsa_v8_evaluation_length.png\n")
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
    print("EVALUATING TRAINED AGENT (V8 - With Safety Mask)")
    print("="*70)
    
    # Use SnakeEnvV8
    env = SnakeEnvV8(render_mode='pygame', cell_size=60)
    
    rewards = []
    lengths = []
    steps_list = []
    
    try:
        # Temporary override epsilon for evaluation
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Render initial state
            env.render(delay=200)
            
            while not done:
                forbidden = get_forbidden_action(env.direction)
                
                # Use get_action which now handles safety internally
                action = agent.get_action(state, forbidden_action=forbidden)
                
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
                    print(f"Episode {ep+1} finished. Length: {len(env.snake)}")
            
            # Collect stats
            current_length = len(env.snake)
            current_steps = env.steps
            
            rewards.append(episode_reward)
            lengths.append(current_length)
            steps_list.append(current_steps)
            
            print(f"Episode {ep+1:2d}: Reward={episode_reward:7.1f}, Length={current_length:2d}, Steps={current_steps:3d}")
    finally:
        agent.epsilon = old_epsilon
        env.close()
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    if rewards:
        print(f"Average Reward: {np.mean(rewards):.1f}")
        print(f"Average Length: {np.mean(lengths):.2f}")
        print(f"Average Steps: {np.mean(steps_list):.1f}")
        print(f"Best Reward: {max(rewards):.1f}")
        print(f"Best Length: {max(lengths)}")
        plot_evaluation_results(rewards, lengths, steps_list)
    print("="*70 + "\n")
    
    return rewards, lengths, steps_list

def train_sarsa_v8(agent=None, episodes=20000):
    env = SnakeEnvV8(render_mode='text')
    if agent is None: agent = SarsaAgentV8()
    
    print(f"Starting V8 training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        forbidden = get_forbidden_action(env.direction)
        
        # get_action now handles safety masking internally
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
    # You can reuse the V7 model since the state space is identical!
    # This allows checking if the Masking fixes the behavior without retraining.
    agent = SarsaAgentV8(load_path="lucas_dev/models/sarsa_v8_50000_episodes.npz")
    
    # If the file doesn't exist, you might need to train:
    # agent = train_sarsa_v8(episodes=50000)
    
    evaluate_agent(agent, num_episodes=5)
