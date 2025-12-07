import numpy as np
import random
from collections import deque
import math
from visualize import SnakeVisualizer

class SnakeEnvV4:
    """
    Snake Game Environment V4: Absolute Position + Direction + Hazard
    
    State Representation:
    1. Head Position (Index 0-63): flattened y*8 + x
    2. Food Position (Index 0-63): flattened y*8 + x
    3. Head Direction (0-3): [Up, Down, Left, Right]
    4. Tail Absolute Direction (0-7): [N, NE, E, SE, S, SW, W, NW]
    5. Hazard Mask (0-15): 4 bits [Up, Down, Left, Right] blocked?
    
    Total States: 64 * 64 * 4 * 8 * 16 = 2,097,152
    
    Actions (Absolute):
    0: UP
    1: DOWN
    2: LEFT
    3: RIGHT
    """
    
    GRID_SIZE = 8
    
    # Base Rewards (Dynamic)
    BASE_REWARD_FOOD = 10
    BASE_REWARD_DEATH = -10
    BASE_REWARD_STEP = -1.0
    
    
    # Absolute Directions / Actions
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
        self.current_score = 0
        
    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.current_score = 0
        
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
        snake_len = len(self.snake)
        
        # Prevent 180 turn
        if (action == self.UP and self.direction == self.DOWN) or \
           (action == self.DOWN and self.direction == self.UP) or \
           (action == self.LEFT and self.direction == self.RIGHT) or \
           (action == self.RIGHT and self.direction == self.LEFT):
            # Invalid move, keep old direction
            action = self.direction
        
        self.direction = action
        
        head_x, head_y = self.snake[0]
        
        if self.direction == self.UP: new_head = (head_x, head_y - 1)
        elif self.direction == self.DOWN: new_head = (head_x, head_y + 1)
        elif self.direction == self.LEFT: new_head = (head_x - 1, head_y)
        else: new_head = (head_x + 1, head_y)
        
        # Dynamic Step Reward
        step_penalty = -1.0 / (0.15 * snake_len + 0.5)
        if step_penalty > -0.05: step_penalty = -0.05
        
        reward = step_penalty
        done = False
        
        # 1. Check Death
        if not self._is_valid_position(new_head) or new_head in self.snake[1:]:
            reward = self.BASE_REWARD_DEATH - (snake_len * 0.5)
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True
            return self.get_state(), reward, done
            
        # 2. Move
        if new_head == self.food:
            reward = self.BASE_REWARD_FOOD + (snake_len * 0.5)
            self.snake.insert(0, new_head)
            self.current_score += 1
            self._spawn_food()
            
            hx, hy = self.snake[0]
            fx, fy = self.food
            self.prev_distance = abs(hx - fx) + abs(hy - fy)
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            
            hx, hy = self.snake[0]
            fx, fy = self.food
            curr_dist = abs(hx - fx) + abs(hy - fy)
            
            if curr_dist < self.prev_distance:
                reward += 0.1
            else:
                reward -= 0.1
            self.prev_distance = curr_dist
            
        self.total_reward += reward
        return self.get_state(), reward, done

    def get_state(self):
        """
        Returns: (head_idx, food_idx, head_dir, tail_dir, hazard_mask)
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food if self.food else (0,0)
        
        # 1. Flattened Head Pos (0-63)
        head_idx = head_y * self.grid_size + head_x
        
        # 2. Flattened Food Pos (0-63)
        food_idx = food_y * self.grid_size + food_x
        
        # 3. Head Direction (0-3)
        head_dir = self.direction
        
        # 4. Tail Absolute Direction (0-7)
        tail_pos = self.snake[-1]
        tail_dir = self._get_absolute_direction(tail_pos, (head_x, head_y))
        
        # 5. Hazard Mask (4 bits)
        trap_mask = 0
        check_dirs = [
            (self.UP, 0, -1, 0),
            (self.DOWN, 0, 1, 1),
            (self.LEFT, -1, 0, 2),
            (self.RIGHT, 1, 0, 3)
        ]
        
        for action, dx, dy, bit in check_dirs:
            nx, ny = head_x + dx, head_y + dy
            pos = (nx, ny)
            
            is_trap = False
            if not self._is_valid_position(pos):
                is_trap = True
            elif pos in self.snake:
                 if pos == self.snake[-1]: 
                     is_trap = False
                 else:
                     is_trap = True
            else:
                reachable = self._flood_fill(pos, self.snake)
                if reachable < len(self.snake):
                    is_trap = True
            
            if is_trap:
                trap_mask |= (1 << bit)
                
        return (head_idx, food_idx, head_dir, tail_dir, trap_mask)

    def _get_absolute_direction(self, target, head):
        if target is None: return 0
        hx, hy = head
        tx, ty = target
        dx = tx - hx
        dy = ty - hy
        
        if dx == 0 and dy == 0: return 0
        
        if dx == 0: return 0 if dy < 0 else 4
        if dy == 0: return 2 if dx > 0 else 6
        if dx > 0: return 1 if dy < 0 else 3
        else: return 7 if dy < 0 else 5

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
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    n_pos not in visited and n_pos not in obstacle_set and n_pos not in self.obstacles):
                    visited.add(n_pos)
                    queue.append(n_pos)
        return count

    def _is_valid_position(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size: return False
        if pos in self.obstacles: return False
        return True
        
    def _spawn_food(self):
        empty = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                p = (x, y)
                if self._is_valid_position(p) and p not in self.snake:
                    empty.append(p)
        if empty: self.food = random.choice(empty)
        else: self.food = None
            
    def render(self, delay=0):
        if self.visualizer:
            self.visualizer.render(self.snake, self.food, self.obstacles, self.steps, self.total_reward, self.direction)
            if delay:
                import pygame
                pygame.time.wait(delay)
    
    def close(self):
        if self.visualizer:
            self.visualizer.close()
