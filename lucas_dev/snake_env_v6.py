
import numpy as np
import random
from collections import deque
from visualize import SnakeVisualizer

class SnakeEnvV6:
    """
    Snake Game Environment V6: "Implicit Planning"
    
    Improvements over V5:
    1. Removed Explicit Length Category (as requested).
    2. Richer Surroundings via "Dynamic Space Levels":
       - Instead of fixed safety thresholds, we use 5 levels of granular accessibility 
         relative to current snake length.
       - This provides clearer gradients of danger for late-game planning.
    
    State Representation:
    1. Food Direction (8 values: 0-7)
    2. Surroundings (4 directions * 5 levels = 625 values):
       - For each direction (Up, Down, Left, Right):
         - 0: Safe (Space >= Length)
         - 1: Cautious (Space >= Length/2)
         - 2: Risky (Space >= Length/4)
         - 3: Dangerous (Space >= 2)
         - 4: Blocked/Dead End (Space <= 1 or Immediate Wall)
       - Encoded as Base-5 integer.
       
    Total States: 8 * 625 = 5000 (< 10,000)
    """
    
    GRID_SIZE = 8
    
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
        self.obstacles = [(2, 2), (5, 2), (2, 5), (5, 5)]
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
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        return self.get_state()
        
    def step(self, action):
        self.steps += 1
        self.direction = action
        
        head_x, head_y = self.snake[0]
        
        if action == self.UP: new_head = (head_x, head_y - 1)
        elif action == self.DOWN: new_head = (head_x, head_y + 1)
        elif action == self.LEFT: new_head = (head_x - 1, head_y)
        else: new_head = (head_x + 1, head_y)
        
        # Using V5 rewards as they were successful
        step_reward = -1.0
        
        reward = step_reward
        done = False
        
        # 1. Check Death
        if not self._is_valid_position(new_head) or new_head in self.snake[1:]:
            # Base -10, plus length penalty
            reward = -10 - (len(self.snake) * 1.0)
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True
            return self.get_state(), reward, done
            
        # 2. Move & Check Food
        if new_head == self.food:
            # Reward grows with snake size
            reward = 10 + (len(self.snake) * 0.5)
            
            self.snake.insert(0, new_head)
            self._spawn_food()
            
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
                reward += 0.2
            else:
                reward += -0.8
            
            self.prev_distance = curr_dist
            
        self.total_reward += reward
        return self.get_state(), reward, done
        
    def get_state(self):
        """
        Returns a tuple: (food_dir, surround_data)
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 1. Food Direction (8 values)
        dx = food_x - head_x
        dy = food_y - head_y
        
        if dx == 0 and dy < 0: f_dir = 0
        elif dx > 0 and dy < 0: f_dir = 1
        elif dx > 0 and dy == 0: f_dir = 2
        elif dx > 0 and dy > 0: f_dir = 3
        elif dx == 0 and dy > 0: f_dir = 4
        elif dx < 0 and dy > 0: f_dir = 5
        elif dx < 0 and dy == 0: f_dir = 6
        else: f_dir = 7
        
        # 2. Surroundings (Base-5 encoded integer)
        # Directions: UP(0), DOWN(1), LEFT(2), RIGHT(3)
        surround_code = 0
        
        check_dirs = [
            (0, -1), # UP
            (0, 1),  # DOWN
            (-1, 0), # LEFT
            (1, 0)   # RIGHT
        ]
        
        current_len = len(self.snake)
        
        for i, (cdx, cdy) in enumerate(check_dirs):
            nx, ny = head_x + cdx, head_y + cdy
            pos = (nx, ny)
            val = 0
            
            # Check Immediate Danger / Wall
            if not self._is_valid_position(pos) or pos in self.snake:
                val = 4 # Blocked
            else:
                # Flood Fill to check space
                # We limit search to snake length to be efficient, as that's our Safe threshold
                space = self._flood_fill(pos, self.snake, limit=current_len + 2)
                
                # Dynamic Levels
                if space >= current_len:
                    val = 0 # Safe
                elif space >= current_len // 2:
                    val = 1 # Cautious
                elif space >= current_len // 4:
                    val = 2 # Risky
                elif space >= 2:
                    val = 3 # Dangerous (but moving)
                else:
                    val = 4 # Dead End (effectively Blocked)
            
            surround_code += val * (5 ** i) # Base 5 encoding
            
        return (f_dir, surround_code)

    def _flood_fill(self, start_pos, obstacles, limit=100):
        """
        BFS to count reachable cells from start_pos
        """
        queue = deque([start_pos])
        visited = set([start_pos])
        count = 0
        
        obstacle_set = set(obstacles)
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count >= limit:
                return count
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                n_pos = (nx, ny)
                
                if (0 <= nx < self.grid_size and 
                    0 <= ny < self.grid_size and 
                    n_pos not in visited and 
                    n_pos not in obstacle_set and 
                    n_pos not in self.obstacles):
                    
                    visited.add(n_pos)
                    queue.append(n_pos)
                    
        return count

    def _is_valid_position(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        if pos in self.obstacles:
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
