import numpy as np
import random
from typing import Tuple, List, Optional

# Try to import visualizer from parent directory or current
try:
    from visualize import SnakeVisualizer
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from visualize import SnakeVisualizer
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False
        print("Warning: pygame not available. Visualization will use text mode only.")

class SnakeEnvV2:
    """
    Snake Game Environment for Reinforcement Learning (V2)
    
    Grid: 8x8
    State: (head_x, head_y, food_x, food_y, danger_mask)
           danger_mask is a 4-bit integer representing obstacles/body in UP, DOWN, LEFT, RIGHT
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    # Grid size
    GRID_SIZE = 8
    
    # Rewards
    REWARD_FOOD = 20      # Increased to encourage eating
    REWARD_DEATH = -10
    REWARD_STEP = -1
    REWARD_CLOSER = 1.1     # Reward for moving closer to food
    REWARD_FARTHER = -2   # Penalty for moving away from food
    
    # Maximum steps per episode
    MAX_STEPS = 1000
    
    def __init__(self, render_mode='pygame', cell_size=60):
        self.grid_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.visualizer = None
        
        if render_mode == 'pygame' and PYGAME_AVAILABLE:
            self.visualizer = SnakeVisualizer(grid_size=self.grid_size, cell_size=cell_size)
        elif render_mode == 'pygame':
            self.render_mode = 'text'
        
        self.snake = []
        self.direction = None
        self.food = None
        self.obstacles = []
        self.steps = 0
        self.total_reward = 0
        self.prev_distance = 0  # Track distance to food
        
        self._init_obstacles()
    
    def _init_obstacles(self):
        self.obstacles = [
            (2, 2), (5, 2), (2, 5), (5, 5)
        ]
    
    def reset(self):
        self.steps = 0
        self.total_reward = 0
        
        center = self.grid_size // 2
        self.direction = random.randint(0, 3)
        
        if self.direction == self.UP:
            self.snake = [(center, center), (center, center + 1), (center, center + 2)]
        elif self.direction == self.DOWN:
            self.snake = [(center, center), (center, center - 1), (center, center - 2)]
        elif self.direction == self.LEFT:
            self.snake = [(center, center), (center + 1, center), (center + 2, center)]
        else:
            self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        
        self._spawn_food()
        
        # Calculate initial distance
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        return self.get_state()
    
    def step(self, action):
        self.steps += 1
        self.direction = action
        
        head_x, head_y = self.snake[0]
        
        if action == self.UP:
            new_head = (head_x, head_y - 1)
        elif action == self.DOWN:
            new_head = (head_x, head_y + 1)
        elif action == self.LEFT:
            new_head = (head_x - 1, head_y)
        else:
            new_head = (head_x + 1, head_y)
        
        reward = self.REWARD_STEP
        done = False
        
        # Check collisions
        if not self._is_valid_position(new_head) or new_head in self.snake[1:]:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Move
        if new_head == self.food:
            reward = self.REWARD_FOOD
            self.snake.insert(0, new_head)
            self._spawn_food()
            
            # Recalculate distance to new food
            head_x, head_y = self.snake[0]
            food_x, food_y = self.food
            self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
            
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            
            # Calculate new distance
            head_x, head_y = self.snake[0]
            food_x, food_y = self.food
            current_distance = abs(head_x - food_x) + abs(head_y - food_y)
            
            # Reward shaping
            if current_distance < self.prev_distance:
                reward += self.REWARD_CLOSER
            else:
                reward += self.REWARD_FARTHER
                
            self.prev_distance = current_distance
            
        self.total_reward += reward
        return self.get_state(), reward, done
    
    def get_state(self):
        """
        New State Representation:
        (head_x, head_y, food_x, food_y, danger_mask)
        
        danger_mask is a 4-bit integer:
        Bit 0 (LSB): Danger UP
        Bit 1: Danger DOWN
        Bit 2: Danger LEFT
        Bit 3: Danger RIGHT
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Calculate danger mask
        # Check all 4 adjacent cells
        # UP (x, y-1)
        up_pos = (head_x, head_y - 1)
        danger_up = 1 if (not self._is_valid_position(up_pos) or up_pos in self.snake) else 0
        
        # DOWN (x, y+1)
        down_pos = (head_x, head_y + 1)
        danger_down = 1 if (not self._is_valid_position(down_pos) or down_pos in self.snake) else 0
        
        # LEFT (x-1, y)
        left_pos = (head_x - 1, head_y)
        danger_left = 1 if (not self._is_valid_position(left_pos) or left_pos in self.snake) else 0
        
        # RIGHT (x+1, y)
        right_pos = (head_x + 1, head_y)
        danger_right = 1 if (not self._is_valid_position(right_pos) or right_pos in self.snake) else 0
        
        # Combine into mask (0-15)
        danger_mask = danger_up | (danger_down << 1) | (danger_left << 2) | (danger_right << 3)
        
        return (head_x, head_y, food_x, food_y, danger_mask)
    
    def _is_valid_position(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def _spawn_food(self):
        empty_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if (self._is_valid_position(pos) and 
                    pos not in self.snake and 
                    pos != self.food):
                    empty_positions.append(pos)
        
        if empty_positions:
            self.food = random.choice(empty_positions)
        else:
            self.food = None

    def render(self, delay=0):
        if self.render_mode == 'pygame' and self.visualizer:
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
            pass # Text render omitted for brevity in V2
            
    def close(self):
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None

    def get_state_space_shape(self):
        # head_x, head_y, food_x, food_y, danger_mask
        return (self.grid_size, self.grid_size, self.grid_size, self.grid_size, 16)
