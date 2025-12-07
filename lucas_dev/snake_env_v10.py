
import numpy as np
import random
from collections import deque
try:
    from visualize import SnakeVisualizer
except ImportError:
    SnakeVisualizer = None

class SnakeEnvV10:
    """
    Snake Game Environment V10: "Hunger & Space Awareness"
    
    Strategies Implemented:
    1. Biological Hunger: Agent dies if it doesn't eat within `max_hunger_steps`.
       - Prevents infinite looping/circling.
    2. Space-Based Safety (Replacing Tail Reachability):
       - Safety is defined by "Open Space > Snake Length".
       - Decouples safety from "following tail", simplifying state and discouraging loops.
    3. Refined Rewards:
       - High food reward.
       - Hunger death penalty.
    
    State Representation: (8 * 256 * 3 = 6144 states)
    1. Food Direction (8 values)
    2. Surroundings (4 directions * 4 levels = 256 values)
       - 0: DEATH (Wall, Body, Obstacle)
       - 1: TRAP (Space available <= Length)
       - 2: SAFE & NEUTRAL (Space > Length, not closer to food)
       - 3: SAFE & CLOSER (Space > Length, closer to food)
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
    
    MAX_STEPS = 1000 # Increased overall limit, but Hunger limit is strict
    
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
        
        # Strategy 1: Hunger
        self.max_hunger_steps = self.GRID_SIZE * self.GRID_SIZE * 2 # 128 steps
        self.current_hunger = 0
        
    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.current_hunger = self.max_hunger_steps # Reset hunger
        
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
        if self.food:
            food_x, food_y = self.food
            self.prev_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        return self.get_state()
        
    def step(self, action):
        self.steps += 1
        self.current_hunger -= 1 # Hunger ticks down
        self.direction = action
        
        head_x, head_y = self.snake[0]
        
        if action == self.UP: new_head = (head_x, head_y - 1)
        elif action == self.DOWN: new_head = (head_x, head_y + 1)
        elif action == self.LEFT: new_head = (head_x - 1, head_y)
        else: new_head = (head_x + 1, head_y)
        
        reward = 0
        done = False
        
        # 1. Check Hunger Death
        if self.current_hunger <= 0:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # 2. Check Physical Death (Wall, Self, Obstacle)
        # Logic: If we eat, tail stays. If we don't eat, tail moves.
        if new_head == self.food:
            unsafe_body = self.snake
        else:
            unsafe_body = self.snake[:-1] # Tail will move away
            
        if not self._is_valid_position(new_head) or new_head in unsafe_body:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True # Timeout (should rarely happen with hunger logic active)
            return self.get_state(), reward, done

        # 3. Check Win (No Food means full board)
        if self.food is None:
            reward = 100 + (len(self.snake) * 20.0)
            done = True
            return self.get_state(), reward, done
            
        # 4. Move & Check Food
        if new_head == self.food:
            reward = self.REWARD_FOOD
            self.snake.insert(0, new_head)
            self.current_hunger = self.max_hunger_steps # RESET HUNGER
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
                # Strategy 3: Check Safety via Open Space
                # We do NOT check "can reach tail". We check "space > length".
                space_count = self._count_space(pos, self.snake)
                
                # Rule: Is space enough to hold us?
                # Add buffer of 2 to be safe.
                is_safe_space = space_count > (len(self.snake) + 2)
                
                if not is_safe_space:
                    val = 1 # Trap
                else:
                    # It is safe because we have room to maneuver
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

    def _count_space(self, start_pos, snake):
        """
        Counts reachable free/open cells from start_pos using BFS.
        Stops early if count exceeds required safety buffer.
        """
        # Treat tail as free space if we move (standard snake logic)
        # But for 'Space', we just treat the current snake configuration as obstacles
        # except the tail which will move.
        obstacles = set(self.snake)
        
        # If start_pos is food, we grow, so tail stays -> Obstacle.
        is_food = (start_pos == self.food)
        if not is_food:
            obstacles.discard(snake[-1]) # Tail will move
            
        if start_pos in obstacles:
            return 0
            
        queue = deque([start_pos])
        visited = set([start_pos])
        obstacle_set = set(self.obstacles)
        
        count = 0
        # Optimization: We only need to know if space > length + buffer
        target_count = len(snake) + 5
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            
            if count >= target_count:
                return count
                
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
