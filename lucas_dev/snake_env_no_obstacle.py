import numpy as np
import random
try:
    from visualize import SnakeVisualizer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from visualize import SnakeVisualizer
    except ImportError:
        SnakeVisualizer = None

class SnakeEnvNoObstacle:
    """
    Snake Game Environment for CNN (No Obstacles)
    
    State Representation:
    Returns a 3-channel grid (C, H, W) -> (3, 8, 8)
    Channel 0: Snake Head (1.0)
    Channel 1: Snake Body (Gradient 1.0 -> 0.1)
    Channel 2: Food (1.0)
    """
    
    GRID_SIZE = 8
    
    # Rewards (Shaped for faster learning)
    REWARD_FOOD = 12.0    # Increased back to make eating worth the risk
    REWARD_DEATH = -10.0
    REWARD_STEP = -0.6    # Significant step penalty to discourage loitering/circling
    REWARD_CLOSER = 0.1   # Small guidance, but not enough to offset step penalty if circling
    
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    MAX_STEPS = 1000
    
    def __init__(self, render_mode='pygame', cell_size=60):
        self.grid_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.visualizer = None
        
        if render_mode == 'pygame':
            try:
                self.visualizer = SnakeVisualizer(grid_size=self.grid_size, cell_size=cell_size)
            except:
                self.render_mode = 'text'
        
        self.snake = []
        self.direction = None
        self.food = None
        self.obstacles = [] # No obstacles
        self.steps = 0
        self.total_reward = 0
        self.prev_distance = 0
        
    def reset(self):
        self.steps = 0
        self.total_reward = 0
        
        center = self.grid_size // 2
        self.direction = random.randint(0, 3)
        
        # Initialize snake (length 3)
        if self.direction == self.UP:
            self.snake = [(center, center), (center, center + 1), (center, center + 2)]
        elif self.direction == self.DOWN:
            self.snake = [(center, center), (center, center - 1), (center, center - 2)]
        elif self.direction == self.LEFT:
            self.snake = [(center, center), (center + 1, center), (center + 2, center)]
        else:
            self.snake = [(center, center), (center - 1, center), (center - 2, center)]
            
        self._spawn_food()
        
        hx, hy = self.snake[0]
        fx, fy = self.food
        self.prev_distance = abs(hx - fx) + abs(hy - fy)
        
        return self.get_state()
        
    def step(self, action):
        self.steps += 1
        self.direction = action
        
        head_x, head_y = self.snake[0]
        
        if action == self.UP: new_head = (head_x, head_y - 1)
        elif action == self.DOWN: new_head = (head_x, head_y + 1)
        elif action == self.LEFT: new_head = (head_x - 1, head_y)
        else: new_head = (head_x + 1, head_y)
        
        reward = self.REWARD_STEP
        done = False
        
        # 1. Check Death
        if not self._is_valid_position(new_head) or new_head in self.snake[1:]:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True
            return self.get_state(), reward, done
            
        # 2. Move
        if new_head == self.food:
            reward = self.REWARD_FOOD
            self.snake.insert(0, new_head)
            self._spawn_food()
            
            # Reset distance
            hx, hy = self.snake[0]
            fx, fy = self.food
            self.prev_distance = abs(hx - fx) + abs(hy - fy)
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            
            # Distance Reward
            hx, hy = self.snake[0]
            fx, fy = self.food
            curr_dist = abs(hx - fx) + abs(hy - fy)
            
            if curr_dist < self.prev_distance:
                reward += self.REWARD_CLOSER
            
            self.prev_distance = curr_dist
            
        self.total_reward += reward
        return self.get_state(), reward, done
        
    def get_state(self):
        """
        Returns (3, 8, 8) grid
        """
        state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Head
        hx, hy = self.snake[0]
        state[0, hy, hx] = -1.0
        
        # Channel 1: Body (Gradient)
        for i, (bx, by) in enumerate(self.snake[1:], start=1):
            # Gradient from near 1.0 (neck) to near 0.0 (tail)
            val = max(0.1, 1.0 - (i * 0.02))
            state[1, by, bx] = -val
            
        # Channel 2: Food
        if self.food:
            fx, fy = self.food
            state[2, fy, fx] = 1.0
            
        return state

    def _is_valid_position(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        return True
        
    def _spawn_food(self):
        empty = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                p = (x, y)
                if self._is_valid_position(p) and p not in self.snake:
                    empty.append(p)
        if empty:
            self.food = random.choice(empty)
        else:
            self.food = None
            
    def render(self, delay=0):
        if self.visualizer:
            self.visualizer.render(self.snake, self.food, self.obstacles, self.steps, self.total_reward, self.direction)
            if delay:
                import pygame
                pygame.time.wait(delay)
    
    def close(self):
        if self.visualizer:
            self.visualizer.close()
