
import numpy as np
import random
from collections import deque
try:
    from visualize import SnakeVisualizer
except ImportError:
    SnakeVisualizer = None

class SnakeEnvV11:
    """
    Snake Game Environment V11: "Hunger + Hybrid Safety"
    
    Changes from V10:
    1. Re-introduced Tail Reachability into Safety check.
       - In V10, Safety was ONLY "Space > Length".
       - In V11, Safety is "Space > Length" OR "Can Reach Tail".
       - This allows the snake to enter tight spaces ("Traps" in V10) if it can cycle its tail, which is crucial for late-game.
    2. Retained Hunger (Strategy 1 from V10):
       - Prevents the agent from abusing the "Tail Reachable" state to simply circle forever.
       - The agent interprets "Safe (Tail)" as "Safe to move, but I still need to eat soon."
    
    State Representation:
       - 0: DEATH
       - 1: TRAP (Space <= Length AND Cannot Reach Tail)
       - 2: SAFE & NEUTRAL ( (Space > Length OR Reach Tail) & Not Closer )
       - 3: SAFE & CLOSER ( (Space > Length OR Reach Tail) & Closer )
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
    
    MAX_STEPS = 1000 
    
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
             self.render_mode = 'text'
        
        self.snake = []
        self.direction = None
        self.food = None
        self.obstacles = [(2, 2), (5, 2), (2, 5), (5, 5)]
        self.steps = 0
        self.total_reward = 0
        self.prev_distance = 0
        
        # Hunger
        self.max_hunger_steps = self.GRID_SIZE * self.GRID_SIZE * 3 # 128 steps
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
        self.current_hunger -= 1 
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
            reward = self.REWARD_DEATH - len(self.snake)
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
        
        # 2. Check Physical Death
        if new_head == self.food:
            unsafe_body = self.snake
        else:
            unsafe_body = self.snake[:-1]
            
        if not self._is_valid_position(new_head) or new_head in unsafe_body:
            reward = self.REWARD_DEATH
            done = True
            self.total_reward += reward
            return self.get_state(), reward, done
            
        if self.steps >= self.MAX_STEPS:
            done = True 
            return self.get_state(), reward, done

        # 3. Check Win
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
            
            if self.food is None:
                reward += 200
                done = True
                return self.get_state(), reward, done
                
            hx, hy = self.snake[0]
            fx, fy = self.food
            self.prev_distance = abs(hx - fx) + abs(hy - fy)
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            hx, hy = self.snake[0]
            
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
        head_x, head_y = self.snake[0]
        
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
        
        # 2. Surroundings
        surround_code = 0
        check_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
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
                # V11 Hybrid Safety Analysis
                space_count, can_reach_tail = self._analyze_safety(pos, self.snake)
                
                # Rule: Safe if (Space > Length+2) OR (Can Reach Tail)
                # This restores the V7 ability to use tail for safety, 
                # but relies on Hunger (in the env) to prevent abuse.
                is_safe = (space_count > (len(self.snake) + 2)) or can_reach_tail
                
                if not is_safe:
                    val = 1 # Trap
                else:
                    new_dist = abs(nx - food_x) + abs(ny - food_y)
                    if new_dist < curr_dist:
                        val = 3 # Safe & Closer
                    else:
                        val = 2 # Safe & Neutral
            
            surround_code += val * (4 ** i)
            
        # 3. Length Category
        length = len(self.snake)
        if length < 8:
            l_cat = 0 
        elif length < 20:
            l_cat = 1 
        else:
            l_cat = 2 
            
        return (f_dir, surround_code, l_cat)

    def _analyze_safety(self, start_pos, snake):
        """
        Returns (space_count: int, can_reach_tail: bool)
        """
        target = snake[-1]
        obstacles = set(self.snake)
        
        is_food = (start_pos == self.food)
        if not is_food and target in obstacles:
            obstacles.remove(target) 
            
        if start_pos in obstacles:
            return 0, False
            
        queue = deque([start_pos])
        visited = set([start_pos])
        obstacle_set = set(self.obstacles)
        
        count = 0
        can_reach_tail = False
        target_count = len(snake) + 5
        
        # If we start AT the tail (rare, but possible if following exactly), we found it.
        if start_pos == target:
            can_reach_tail = True

        while queue:
            cx, cy = queue.popleft()
            count += 1
            
            if (cx, cy) == target:
                can_reach_tail = True
            
            # If we found both enough space AND the tail, early exit?
            # Actually, if we found tail, we are safe regardless of space count (mostly).
            # But let's let it count up to target_count to be sure about "Space" metric if needed.
            # Optimization: If both conditions met (or if tailored for boolean logic):
            if count >= target_count and can_reach_tail:
                break
            if count >= target_count and not can_reach_tail:
                # We have enough space, maybe we don't care about tail?
                # But we should return consistent data.
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
                    
        return count, can_reach_tail

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
