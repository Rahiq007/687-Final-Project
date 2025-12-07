import numpy as np
import random
from collections import deque
from visualize import SnakeVisualizer

class SnakeEnvV3:
    """
    Snake Game Environment V3: Planning & Trap Awareness
    
    State Representation (Compact):
    1. Food Direction (8 values: 0-7)
    2. Danger Mask (16 values: 0-15) - Immediate collision
    3. Trap Mask (16 values: 0-15) - Dead end detection (Planning)
    
    Total States: 8 * 16 * 16 = 2048
    """
    
    GRID_SIZE = 8
    
    # Rewards
    REWARD_FOOD = 10
    REWARD_DEATH = -7
    REWARD_STEP = -0.6
    REWARD_CLOSER = 0.5
    REWARD_FARTHER = -0.2
    
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
            else:
                reward += self.REWARD_FARTHER
            self.prev_distance = curr_dist
            
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
                # If it's an immediate danger, it's also a "trap" in the sense you can't go there
                trap_mask |= (1 << bit)
            else:
                # Check Trap (Flood Fill)
                # If we move to 'pos', how many cells can we reach?
                # We simulate the move: head becomes 'pos', tail moves (unless we eat, but let's assume worst case: we don't eat immediately or we grow)
                # Conservative check: Treat current snake body as static obstacles for the flood fill
                reachable = self._flood_fill(pos, self.snake)
                
                # If reachable area is smaller than current snake length, it's a trap!
                if reachable < len(self.snake):
                    trap_mask |= (1 << bit)
                    
        return (f_dir, danger_mask, trap_mask)

    def _flood_fill(self, start_pos, obstacles):
        """
        BFS to count reachable cells from start_pos
        """
        queue = deque([start_pos])
        visited = set([start_pos])
        count = 0
        
        # Convert obstacles to set for O(1) lookup
        # Note: We must treat the snake body as obstacles
        obstacle_set = set(obstacles)
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            
            # Optimization: If we already found enough space, stop
            if count > len(self.snake) * 2: # Heuristic: if we have 2x length space, we are safe
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
