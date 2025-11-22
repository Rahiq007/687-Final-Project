import numpy as np
import random
from typing import Tuple, List, Optional

# Import visualizer (optional - only if pygame is available)
try:
    from visualize import SnakeVisualizer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Visualization will use text mode only.")

class SnakeEnv:
    """
    Snake Game Environment for Reinforcement Learning
    
    Grid: 8x8
    State: (head_x, head_y, food_x, food_y, direction)
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    
    Rewards:
        +10 for eating food
        -20 for death (wall, self-collision, obstacle)
        -1 for each step
    """
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    # Grid size
    GRID_SIZE = 8
    
    # Rewards
    REWARD_FOOD = 10
    REWARD_DEATH = -20
    REWARD_STEP = -1
    
    # Maximum steps per episode (prevent infinite loops)
    MAX_STEPS = 1000
    
    def __init__(self, render_mode='pygame', cell_size=60):
        """
        Initialize the Snake environment
        
        Args:
            render_mode: 'text' for console output, 'pygame' for graphical window (default: 'pygame')
            cell_size: Size of each cell in pixels for pygame rendering (default 60)
        """
        # Grid dimensions
        self.grid_size = self.GRID_SIZE
        
        # Render settings
        self.render_mode = render_mode
        self.visualizer = None
        
        # Initialize pygame visualizer if requested
        if render_mode == 'pygame' and PYGAME_AVAILABLE:
            self.visualizer = SnakeVisualizer(grid_size=self.grid_size, cell_size=cell_size)
            print("Pygame visualizer enabled!")
        elif render_mode == 'pygame' and not PYGAME_AVAILABLE:
            print("Pygame not available. Falling back to text mode.")
            self.render_mode = 'text'
        
        # Snake properties
        self.snake = []  # List of (x, y) positions, head is at index 0
        self.direction = None  # Current direction (0-3)
        
        # Food position
        self.food = None  # (x, y) position
        
        # Obstacles (we'll define fixed positions)
        self.obstacles = []  # List of (x, y) positions
        
        # Episode tracking
        self.steps = 0
        self.total_reward = 0
        
        # Define obstacles (fixed positions)
        self._init_obstacles()
        
        print("Snake Environment initialized!")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Render mode: {self.render_mode}")
        print(f"Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT")
        print(f"State space size: {self.grid_size} x {self.grid_size} x {self.grid_size} x {self.grid_size} x 4")
    
    def _init_obstacles(self):
        """Initialize fixed obstacle positions"""
        # Place 4 obstacles at fixed positions
        # Let's place them strategically to make the game interesting but not too hard
        self.obstacles = [
            (2, 2),  # Top-left region
            (5, 2),  # Top-right region
            (2, 5),  # Bottom-left region
            (5, 5),  # Bottom-right region
        ]
        print(f"Obstacles placed at: {self.obstacles}")
    
    def reset(self) -> Tuple[int, int, int, int, int]:
        """
        Reset the environment to start a new episode
        
        Returns:
            state: (head_x, head_y, food_x, food_y, direction)
        """
        # Reset step counter and total reward
        self.steps = 0
        self.total_reward = 0
        
        # Initialize snake at center with length 3
        center = self.grid_size // 2  # This will be 4 for 8x8 grid
        
        # Choose random initial direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        self.direction = random.randint(0, 3)
        
        # Create snake based on initial direction
        # Snake will be: [head, body1, body2]
        if self.direction == self.UP:
            # Snake facing up, body extends downward
            self.snake = [(center, center), (center, center + 1), (center, center + 2)]
        elif self.direction == self.DOWN:
            # Snake facing down, body extends upward
            self.snake = [(center, center), (center, center - 1), (center, center - 2)]
        elif self.direction == self.LEFT:
            # Snake facing left, body extends rightward
            self.snake = [(center, center), (center + 1, center), (center + 2, center)]
        else:  # self.direction == self.RIGHT
            # Snake facing right, body extends leftward
            self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        
        # Spawn initial food
        self._spawn_food()
        
        # Return initial state
        return self.get_state()
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int, int, int], int, bool]:
        """
        Take an action in the environment
        
        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            
        Returns:
            next_state: (head_x, head_y, food_x, food_y, direction)
            reward: The reward received
            done: Whether the episode is finished
        """
        # Increment step counter
        self.steps += 1
        
        # Update direction based on action
        self.direction = action
        
        # Get current head position
        head_x, head_y = self.snake[0]
        
        # Calculate new head position based on action
        if action == self.UP:
            new_head = (head_x, head_y - 1)
        elif action == self.DOWN:
            new_head = (head_x, head_y + 1)
        elif action == self.LEFT:
            new_head = (head_x - 1, head_y)
        else:  # action == self.RIGHT
            new_head = (head_x + 1, head_y)
        
        # Initialize reward and done flag
        reward = self.REWARD_STEP  # Default step penalty
        done = False
        
        # Check for collisions (death conditions)
        new_x, new_y = new_head
        
        # Check wall collision
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Check obstacle collision
        if new_head in self.obstacles:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Check self-collision (hitting own body)
        if new_head in self.snake[1:]:  # Don't check head against itself
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Check if max steps reached
        if self.steps >= self.MAX_STEPS:
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # No collision - snake can move
        # Check if food is eaten
        if new_head == self.food:
            # Snake ate food!
            reward = self.REWARD_FOOD
            # Grow snake: add new head, keep all body segments
            self.snake.insert(0, new_head)
            # Spawn new food
            self._spawn_food()
        else:
            # Normal move: add new head, remove tail
            self.snake.insert(0, new_head)
            self.snake.pop()  # Remove last segment (tail)
        
        # Update total reward
        self.total_reward += reward
        
        # Return next state, reward, and done flag
        return self.get_state(), reward, done
    
    def get_state(self) -> Tuple[int, int, int, int, int]:
        """
        Get the current state representation
        
        Returns:
            state: (head_x, head_y, food_x, food_y, direction)
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return (head_x, head_y, food_x, food_y, self.direction)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (inside grid and not an obstacle)
        
        Args:
            pos: (x, y) position
            
        Returns:
            True if valid, False otherwise
        """
        x, y = pos
        # Check if inside grid
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        # Check if it's an obstacle
        if pos in self.obstacles:
            return False
        return True
    
    def _spawn_food(self):
        """Spawn food at a random empty position"""
        # Get all valid empty positions
        empty_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                # Check if position is valid and not occupied
                if (self._is_valid_position(pos) and 
                    pos not in self.snake and 
                    pos != self.food):
                    empty_positions.append(pos)
        
        # Randomly select one empty position
        if empty_positions:
            self.food = random.choice(empty_positions)
        else:
            # This shouldn't happen unless snake fills the entire grid
            self.food = None
    
    def render(self, delay=0):
        """
        Render the current state of the game
        
        Args:
            delay: Milliseconds to wait after rendering (for pygame mode)
        """
        if self.render_mode == 'pygame' and self.visualizer:
            # Use pygame visualization
            self.visualizer.render(
                snake=self.snake,
                food=self.food,
                obstacles=self.obstacles,
                steps=self.steps,
                score=self.total_reward,
                direction=self.direction
            )
            if delay > 0:
                import pygame
                pygame.time.wait(delay)
        else:
            # Use text visualization
            self._render_text()
    
    def _render_text(self):
        """Print the current state of the game (text-based)"""
        # Create empty grid
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place obstacles
        for x, y in self.obstacles:
            grid[y][x] = 'X'
        
        # Place snake body
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 'H'  # Head
            else:
                grid[y][x] = 'S'  # Body
        
        # Place food
        if self.food:
            fx, fy = self.food
            grid[fy][fx] = 'F'
        
        # Print grid
        print("\n" + "=" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("=" * (self.grid_size * 2 + 1))
        print(f"Steps: {self.steps} | Direction: {['UP', 'DOWN', 'LEFT', 'RIGHT'][self.direction]} | Snake Length: {len(self.snake)}")
        print(f"State: {self.get_state()}")
    
    def close(self):
        """Close the environment and cleanup"""
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None
    
    # ==================== Helper Methods for RL Algorithms ====================
    
    def get_action_space_size(self) -> int:
        """
        Get the number of possible actions
        
        Returns:
            4 (UP, DOWN, LEFT, RIGHT)
        """
        return 4
    
    def get_state_space_size(self) -> int:
        """
        Get the total number of possible states
        
        Returns:
            grid_size^4 * 4 (x, y, food_x, food_y, direction)
            For 8x8 grid: 8*8*8*8*4 = 16,384 states
        """
        return self.grid_size ** 4 * 4
    
    def get_state_bounds(self) -> dict:
        """
        Get the bounds for each component of the state
        
        Returns:
            Dictionary with min/max values for each state component
        """
        return {
            'head_x': (0, self.grid_size - 1),
            'head_y': (0, self.grid_size - 1),
            'food_x': (0, self.grid_size - 1),
            'food_y': (0, self.grid_size - 1),
            'direction': (0, 3)
        }
    
    def get_grid_size(self) -> int:
        """Get the size of the grid (8 for 8x8)"""
        return self.grid_size
    
    def get_obstacles(self) -> List[Tuple[int, int]]:
        """Get the list of obstacle positions"""
        return self.obstacles.copy()
    
    def get_current_snake_length(self) -> int:
        """Get the current length of the snake"""
        return len(self.snake)
    
    def get_episode_info(self) -> dict:
        """
        Get information about the current episode
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'snake_length': len(self.snake),
            'food_position': self.food,
            'snake_head': self.snake[0] if self.snake else None
        }


if __name__ == "__main__":
    print("="*60)
    print("SNAKE ENVIRONMENT - COMPREHENSIVE TEST")
    print("="*60)
    
    # Test with text mode first
    print("\n" + "="*60)
    print("MODE 1: Text Rendering")
    print("="*60)
    
    env = SnakeEnv(render_mode='text')
    state = env.reset()
    print(f"Initial state: {state}")
    env.render()
    
    # Take a few steps
    print("\nTaking 3 steps...")
    for i in range(3):
        action = env.direction
        next_state, reward, done = env.step(action)
        print(f"\nStep {i+1}: Reward={reward}")
        env.render()
        if done:
            break
    
    # Now test with pygame if available
    if PYGAME_AVAILABLE:
        print("\n" + "="*60)
        print("MODE 2: Pygame Visualization")
        print("="*60)
        print("Creating environment with pygame visualization...")
        print("Close the window to continue...")
        
        env_pygame = SnakeEnv(render_mode='pygame', cell_size=60)
        state = env_pygame.reset()
        env_pygame.render()
        
        # Run a demo episode
        import pygame
        running = True
        episode_done = False
        
        while running and not episode_done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Take action (move in current direction)
            action = env_pygame.direction
            next_state, reward, done = env_pygame.step(action)
            
            # Render with delay
            env_pygame.render(delay=200)
            
            if done:
                print(f"\nEpisode ended! Total reward: {env_pygame.total_reward}")
                episode_done = True
                pygame.time.wait(2000)  # Show final state for 2 seconds
        
        env_pygame.close()
        print("Pygame visualization closed.")
    else:
        print("\n" + "="*60)
        print("Pygame not available - install with: pip install pygame")
        print("="*60)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)