import numpy as np
import random
import math
from collections import deque
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

class SnakeEnvDynamic:
    """
    Snake Game Environment with Dynamic Rewards
    
    State Representation (Compact):
    1. Food Direction (8 values: 0-7)
    2. Danger Mask (16 values: 0-15) - Immediate collision
    3. Trap Mask (16 values: 0-15) - Dead end detection (Planning)
    
    Total States: 8 * 16 * 16 = 2048
    """
    
    GRID_SIZE = 8
    
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    MAX_STEPS = 1000 # Step limit
    
    def __init__(self, render_mode='pygame', cell_size=60):
        self.grid_size = self.GRID_SIZE
        self.capacity = self.grid_size * self.grid_size # Max possible snake size
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
        self.reward_step_counter = 0
        self.prev_head = None
        
    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.reward_step_counter = 0
        
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
        self.prev_head = self.snake[0]
        
        return self.get_state()
        
    def step(self, action):
        self.steps += 1
        self.reward_step_counter += 1
        self.direction = action
        
        head_x, head_y = self.snake[0]
        self.prev_head = (head_x, head_y)
        
        if action == self.UP: new_head = (head_x, head_y - 1)
        elif action == self.DOWN: new_head = (head_x, head_y + 1)
        elif action == self.LEFT: new_head = (head_x - 1, head_y)
        else: new_head = (head_x + 1, head_y)
        
        reward = 0.0
        done = False
        
        # 0. Check Victory
        if len(self.snake) == self.capacity:
            reward = self.capacity * 0.1
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        # 1. Check Step Limit
        if self.reward_step_counter > self.MAX_STEPS:
            self.reward_step_counter = 0
            done = True # Timeout treated as death/done
            # Usually timeout doesn't get the death penalty, but let's see. 
            # The snippet implies "Step limit reached, game over" then falls through? 
            # No, if done is set, it might hit the next block.
            # In snippet: if reward_step_counter > limit: done=True. Then if done: penalty.
            # So timeout gets death penalty.
        
        # 2. Check Death (Collision)
        if done or not self._is_valid_position(new_head) or new_head in self.snake[1:]:
            done = True
            # Penalty based on snake size
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth)
            # Using capacity for max_growth and grid_size in snippet context
            exponent = (self.capacity - len(self.snake)) / self.capacity
            reward = - math.pow(self.capacity, exponent)
            reward = reward * 0.1
            self.total_reward += reward
            return self.get_state(), reward, done
            
        # 3. Move & Check Food
        food_obtained = False
        if new_head == self.food:
            food_obtained = True
            self.snake.insert(0, new_head)
            self._spawn_food()
            self.reward_step_counter = 0
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            
        # 4. Calculate Reward
        if food_obtained:
            # Reward boost on snake size
            reward = len(self.snake) / self.capacity
        else:
            # Distance guidance
            # Euclidean distance
            head_pos = np.array(new_head)
            prev_pos = np.array(self.prev_head)
            food_pos = np.array(self.food)
            
            curr_dist = np.linalg.norm(head_pos - food_pos)
            prev_dist = np.linalg.norm(prev_pos - food_pos)
            
            if curr_dist < prev_dist:
                reward = 1 / len(self.snake)
            else:
                reward = -0.5 / len(self.snake)
            
            reward = reward * 0.1
            
        self.total_reward += reward
        return self.get_state(), reward, done
        
    def get_state(self):
        """
        Returns: (food_dir, danger_mask, trap_mask)
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 1. Food Direction (8 values)
        dx = food_x - head_x
        dy = food_y - head_y
        
        if dx == 0 and dy < 0: f_dir = 0 # N
        elif dx > 0 and dy < 0: f_dir = 1 # NE
        elif dx > 0 and dy == 0: f_dir = 2 # E
        elif dx > 0 and dy > 0: f_dir = 3 # SE
        elif dx == 0 and dy > 0: f_dir = 4 # S
        elif dx < 0 and dy > 0: f_dir = 5 # SW
        elif dx < 0 and dy == 0: f_dir = 6 # W
        else: f_dir = 7 # NW
        
        # 2. Danger Mask (Immediate)
        # 3. Trap Mask (Planning - BFS)
        danger_mask = 0
        trap_mask = 0
        
        # Check all 4 directions
        directions = [
            (0, -1, 0), # UP (bit 0)
            (0, 1, 1),  # DOWN (bit 1)
            (-1, 0, 2), # LEFT (bit 2)
            (1, 0, 3)   # RIGHT (bit 3)
        ]
        
        for dx, dy, bit in directions:
            nx, ny = head_x + dx, head_y + dy
            pos = (nx, ny)
            
            # Check Danger
            if not self._is_valid_position(pos) or pos in self.snake:
                danger_mask |= (1 << bit)
                trap_mask |= (1 << bit)
            else:
                # Check Trap (Flood Fill)
                reachable = self._flood_fill(pos, self.snake)
                if reachable < len(self.snake):
                    trap_mask |= (1 << bit)
                    
        return (f_dir, danger_mask, trap_mask)

    def _flood_fill(self, start_pos, obstacles):
        queue = deque([start_pos])
        visited = set([start_pos])
        count = 0
        obstacle_set = set(obstacles)
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count > len(self.snake) * 2: 
                return count
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                n_pos = (nx, ny)
                
                if (0 <= nx < self.grid_size and 
                    0 <= ny < self.grid_size and 
                    n_pos not in visited and 
                    n_pos not in obstacle_set):
                    
                    visited.add(n_pos)
                    queue.append(n_pos)
                    
        return count

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
