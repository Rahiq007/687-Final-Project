"""
Snake Game Environment - WITH BODY AWARENESS
This version adds danger signals so the snake can see its body!

Grid: 8x8
State: (head_x, head_y, food_x, food_y, direction, danger_straight, danger_left, danger_right)
State Space: 131,072 states

Key Feature: Snake can now avoid its own body!

Author: Rahiq Majeed
Course: COMPSCI 687 - Fall 2025
"""

import numpy as np
import random
from typing import Tuple

try:
    from visualize import SnakeVisualizer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class SnakeEnv:
    """Snake Game with Body Awareness through Danger Signals"""
    
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    GRID_SIZE = 8
    
    REWARD_FOOD = 10
    REWARD_DEATH = -20
    REWARD_STEP = -1
    
    MAX_STEPS = 1000
    
    def __init__(self, render_mode='pygame', cell_size=60):
        self.grid_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.visualizer = None
        
        if render_mode == 'pygame' and PYGAME_AVAILABLE:
            self.visualizer = SnakeVisualizer(grid_size=self.grid_size, cell_size=cell_size)
        elif render_mode == 'pygame' and not PYGAME_AVAILABLE:
            self.render_mode = 'text'
        
        self.snake = []
        self.direction = None
        self.food = None
        self.obstacles = []
        self.steps = 0
        self.total_reward = 0
        
        self._init_obstacles()
        
        print("Snake Environment - WITH BODY AWARENESS")
        print(f"Grid: {self.grid_size}x{self.grid_size}")
        print(f"State: (head_x, head_y, food_x, food_y, direction, danger_straight, danger_left, danger_right)")
        print(f"State space: {self.grid_size**4 * 4 * 8:,} states")
        print(f"Max possible length: {self.grid_size**2 - len(self.obstacles)}")
    
    def _init_obstacles(self):
        self.obstacles = [(2, 2), (5, 2), (2, 5), (5, 5)]
    
    def reset(self) -> Tuple:
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
        return self.get_state()
    
    def _is_danger(self, x, y) -> bool:
        """Check if position is dangerous (wall, obstacle, or snake body)"""
        # Wall
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Obstacle
        if (x, y) in self.obstacles:
            return True
        # Snake body (excluding tail since it will move)
        if (x, y) in self.snake[:-1]:
            return True
        return False
    
    def _get_danger_signals(self) -> Tuple[int, int, int]:
        """
        Get danger in 3 directions relative to current heading.
        Returns: (danger_straight, danger_left, danger_right)
        Each is 1 if danger, 0 if safe
        """
        head_x, head_y = self.snake[0]
        
        # Get positions for straight, left, right relative to current direction
        if self.direction == self.UP:
            straight = (head_x, head_y - 1)
            left = (head_x - 1, head_y)
            right = (head_x + 1, head_y)
        elif self.direction == self.DOWN:
            straight = (head_x, head_y + 1)
            left = (head_x + 1, head_y)
            right = (head_x - 1, head_y)
        elif self.direction == self.LEFT:
            straight = (head_x - 1, head_y)
            left = (head_x, head_y + 1)
            right = (head_x, head_y - 1)
        else:  # RIGHT
            straight = (head_x + 1, head_y)
            left = (head_x, head_y - 1)
            right = (head_x, head_y + 1)
        
        danger_straight = 1 if self._is_danger(*straight) else 0
        danger_left = 1 if self._is_danger(*left) else 0
        danger_right = 1 if self._is_danger(*right) else 0
        
        return danger_straight, danger_left, danger_right
    
    def get_state(self) -> Tuple:
        """Get current state with danger awareness"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food if self.food else (0, 0)
        danger_s, danger_l, danger_r = self._get_danger_signals()
        
        return (head_x, head_y, food_x, food_y, self.direction, danger_s, danger_l, danger_r)
    
    def step(self, action: int) -> Tuple:
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
        
        new_x, new_y = new_head
        
        # Check wall
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Check obstacle
        if new_head in self.obstacles:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Check self-collision
        if new_head in self.snake[:-1]:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Check max steps
        if self.steps >= self.MAX_STEPS:
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check food
        if new_head == self.food:
            reward = self.REWARD_FOOD
            self._spawn_food()
        else:
            self.snake.pop()
        
        self.total_reward += reward
        return self.get_state(), reward, done
    
    def _spawn_food(self):
        empty = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if pos not in self.obstacles and pos not in self.snake:
                    empty.append(pos)
        
        if empty:
            self.food = random.choice(empty)
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
            self._render_text()
    
    def _render_text(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for x, y in self.obstacles:
            grid[y][x] = 'X'
        
        for i, (x, y) in enumerate(self.snake):
            grid[y][x] = 'H' if i == 0 else 'S'
        
        if self.food:
            grid[self.food[1]][self.food[0]] = 'F'
        
        print("\n" + "=" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("=" * (self.grid_size * 2 + 1))
        
        state = self.get_state()
        print(f"Length: {len(self.snake)} | Steps: {self.steps}")
        print(f"Dangers - Straight: {state[5]}, Left: {state[6]}, Right: {state[7]}")
    
    def close(self):
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None
    
    def get_episode_info(self) -> dict:
        return {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'snake_length': len(self.snake),
            'food_position': self.food,
            'snake_head': self.snake[0] if self.snake else None
        }