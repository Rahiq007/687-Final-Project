
import numpy as np
import random
from collections import deque
try:
    from visualize import SnakeVisualizer
except ImportError:
    SnakeVisualizer = None

class SnakeEnvV9:
    """
    Snake Game Environment V9: "Safety & Length Awareness"
    
    Improvements:
    1. Includes Snake Length in State: Helps agent distinguish strategies for small vs large snakes.
    2. Refined Rewards: Heavily punished death to prioritize survival. Reduced 'moving away' penalty.
    3. Surrounding State:
       - 0: DEATH (Wall, Body, Obstacle)
       - 1: TRAP (Space available < Length OR cannot reach tail)
       - 2: SAFE & NEUTRAL (Can reach tail, not closer to food)
       - 3: SAFE & CLOSER (Can reach tail, closer to food)
    
    State Representation: (8 * 256 * 3 = 6144 states)
    1. Food Direction (8 values)
    2. Surroundings (4 directions * 4 levels = 256 values)
    3. Length Category (3 values): Small, Medium, Large
    """
    
    # Global Static Reward Baseline
    REWARD_FOOD = 10.0
    REWARD_STEP_CLOSER = -0.2
    REWARD_STEP_FURTHER = -0.8
    REWARD_DEATH = -7.0
    
    GRID_SIZE = 8
    
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    MAX_STEPS = 750
    
    def __init__(self, render_mode='pygame', cell_size=60):
        self.grid_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.visualizer = None
        
        if render_mode == 'pygame' and SnakeVisualizer:
            try:
                self.visualizer = SnakeVisualizer(grid_size=self.grid_size, cell_size=cell_size)
            except:
                self.render_mode = 'text'
        elif render_mode == 'pygame' and not SnakeVisualizer:
             print("Warning: Pygame/Visualizer not found. Switching to text mode.")
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
        
        # Rewards
        step_reward = -0.1 # Slight negative for time passing
        
        reward = step_reward
        done = False
        
        # 1. Check Death
        # Logic: If we eat, tail stays. If we don't eat, tail moves.
        if new_head == self.food:
            unsafe_body = self.snake
        else:
            unsafe_body = self.snake[:-1] # Tail will move away
            
        if not self._is_valid_position(new_head) or new_head in unsafe_body:
            # Heavy penalty for death
            reward = self.REWARD_DEATH - (len(self.snake) * 1.0)
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True
            return self.get_state(), reward, done

        # 2. Check Win (No Food means full board)
        if self.food is None:
            reward = 100 + (len(self.snake) * 20.0)
            done = True
            return self.get_state(), reward, done
            
        # 3. Move & Check Food
        if new_head == self.food:
            # reward = self.REWARD_FOOD + (len(self.snake) * 0.5) # Significant food reward
            reward = self.REWARD_FOOD
            self.snake.insert(0, new_head)
            self._spawn_food()
            
            # Check for Win after eating
            if self.food is None:
                reward += 200 # Bonus for clearing board
                done = True
                return self.get_state(), reward, done
                
            hx, hy = self.snake[0]
            fx, fy = self.food
            self.prev_distance = abs(hx - fx) + abs(hy - fy)
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            hx, hy = self.snake[0]
            
            # Verify food exists
            if self.food is not None:
                fx, fy = self.food
                curr_dist = abs(hx - fx) + abs(hy - fy)
                if curr_dist < self.prev_distance:
                    reward = self.REWARD_STEP_CLOSER
                else:
                    reward = self.REWARD_STEP_FURTHER
                self.prev_distance = curr_dist
            
        self.total_reward += reward
        return self.get_state(), reward, done
        
    def get_state(self):
        """
        Returns a tuple: (food_dir, surround_code, length_cat)
        """
        head_x, head_y = self.snake[0]
        
        # Handle Win/No Food State gracefully
        if self.food is None:
            return (0, 0, 2)
            
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
        
        # 2. Surroundings (Base-4 encoded integer)
        surround_code = 0
        check_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)] # UP, DOWN, LEFT, RIGHT
        
        curr_dist = abs(head_x - food_x) + abs(head_y - food_y)
        
        for i, (cdx, cdy) in enumerate(check_dirs):
            nx, ny = head_x + cdx, head_y + cdy
            pos = (nx, ny)
            val = 0
            
            # Check Immediate Death
            if not self._is_valid_position(pos):
                val = 0
            elif pos in self.snake and pos != self.snake[-1]:
                val = 0
            else:
                # Check Safety / Trap
                # Determine "Space Available" via flood fill or reachability
                can_reach_tail, space_count = self._analyze_safety(pos, self.snake)
                
                if not can_reach_tail and space_count < len(self.snake):
                    val = 1 # Trap
                else:
                    # It is safe (either reach tail OR huge space)
                    # Check Food Progress
                    new_dist = abs(nx - food_x) + abs(ny - food_y)
                    if new_dist < curr_dist:
                        val = 3 # Safe & Closer
                    else:
                        val = 2 # Safe & Neutral
            
            surround_code += val * (4 ** i)
            
        # 3. Length Category
        length = len(self.snake)
        if length < 8:
            l_cat = 0 # Small
        elif length < 20:
            l_cat = 1 # Medium
        else:
            l_cat = 2 # Large
            
        return (f_dir, surround_code, l_cat)

    def _analyze_safety(self, start_pos, snake):
        """
        Returns (can_reach_tail: bool, reachable_count: int)
        """
        target = snake[-1]
        
        obstacles = set(self.snake)
        
        # Only treat tail as free if we are NOT eating (moving to food)
        # If start_pos is food, we grow, so tail stays -> Obstacle.
        is_food = (start_pos == self.food)
        
        if not is_food and target in obstacles:
            obstacles.remove(target) # Tail will move
        
        if start_pos in obstacles:
            return False, 0
            
        queue = deque([start_pos])
        visited = set([start_pos])
        obstacle_set = set(self.obstacles)
        
        can_reach_tail = False
        count = 0
        max_search = 100 # Optimization cap, board is 64 anyway
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            
            if (cx, cy) == target:
                can_reach_tail = True
            
            if count >= max_search:
                break
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                n_pos = (nx, ny)
                
                if (0 <= nx < self.grid_size and 
                    0 <= ny < self.grid_size and 
                    n_pos not in obstacles and
                    n_pos not in obstacle_set and
                    n_pos not in visited):
                    
                    visited.add(n_pos)
                    queue.append(n_pos)
                    
        return can_reach_tail, count

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
