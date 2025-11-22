import pygame
import sys

class SnakeVisualizer:
    """
    Pygame-based visualization for Snake RL Environment
    """
    
    # Colors (RGB)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (220, 20, 60)          # Food
    GREEN = (50, 205, 50)        # Snake head
    DARK_GREEN = (34, 139, 34)   # Snake body
    GRAY = (128, 128, 128)       # Obstacles
    LIGHT_GRAY = (211, 211, 211) # Grid lines
    BLUE = (70, 130, 180)        # Background
    
    def __init__(self, grid_size=8, cell_size=60):
        """
        Initialize the visualizer
        
        Args:
            grid_size: Size of the grid (default 8x8)
            cell_size: Size of each cell in pixels (default 60)
        """
        pygame.init()
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Calculate window size
        self.grid_width = grid_size * cell_size
        self.grid_height = grid_size * cell_size
        self.info_height = 80  # Space for score/info at bottom
        
        self.width = self.grid_width
        self.height = self.grid_height + self.info_height
        
        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake RL Environment")
        
        # Font for text
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        print("Pygame visualizer initialized!")
        print(f"Window size: {self.width}x{self.height}")
    
    def draw_grid(self):
        """Draw the grid lines"""
        for x in range(0, self.grid_width, self.cell_size):
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (x, 0), (x, self.grid_height))
        for y in range(0, self.grid_height, self.cell_size):
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (0, y), (self.grid_width, y))
    
    def draw_cell(self, x, y, color):
        """
        Draw a colored cell at grid position (x, y)
        
        Args:
            x, y: Grid coordinates (0 to grid_size-1)
            color: RGB color tuple
        """
        rect = pygame.Rect(
            x * self.cell_size + 2,
            y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, color, rect)
    
    def draw_circle_cell(self, x, y, color):
        """
        Draw a colored circle at grid position (x, y) - used for food
        
        Args:
            x, y: Grid coordinates (0 to grid_size-1)
            color: RGB color tuple
        """
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 4
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
    
    def render(self, snake, food, obstacles, steps, score, direction):
        """
        Render the current state of the environment
        
        Args:
            snake: List of (x, y) positions, head at index 0
            food: (x, y) position of food
            obstacles: List of (x, y) positions
            steps: Current step count
            score: Current score/total reward
            direction: Current direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
        
        # Clear screen with background color
        self.screen.fill(self.BLUE)
        
        # Draw grid area background
        pygame.draw.rect(self.screen, self.WHITE, (0, 0, self.grid_width, self.grid_height))
        
        # Draw grid lines
        self.draw_grid()
        
        # Draw obstacles
        for x, y in obstacles:
            self.draw_cell(x, y, self.GRAY)
        
        # Draw snake body (all segments except head)
        for i, (x, y) in enumerate(snake[1:]):
            self.draw_cell(x, y, self.DARK_GREEN)
        
        # Draw snake head
        if snake:
            head_x, head_y = snake[0]
            self.draw_cell(head_x, head_y, self.GREEN)
        
        # Draw food as a circle
        if food:
            food_x, food_y = food
            self.draw_circle_cell(food_x, food_y, self.RED)
        
        # Draw info panel at bottom
        info_y = self.grid_height + 10
        
        # Direction names
        direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        direction_str = direction_names[direction] if 0 <= direction <= 3 else 'NONE'
        
        # Create info text
        info_text = f"Steps: {steps}  |  Score: {score:.1f}  |  Length: {len(snake)}  |  Dir: {direction_str}"
        text_surface = self.small_font.render(info_text, True, self.BLACK)
        text_rect = text_surface.get_rect(center=(self.width // 2, info_y + 25))
        self.screen.blit(text_surface, text_rect)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate (30 FPS)
        self.clock.tick(30)
    
    def close(self):
        """Close the visualizer"""
        pygame.quit()
        print("Visualizer closed.")


if __name__ == "__main__":
    # Test the visualizer
    print("Testing Snake Visualizer...")
    
    viz = SnakeVisualizer(grid_size=8, cell_size=60)
    
    # Create test data
    snake = [(4, 4), (4, 5), (4, 6)]
    food = (6, 2)
    obstacles = [(2, 2), (5, 2), (2, 5), (5, 5)]
    
    print("\nDisplaying test visualization...")
    print("Close the window to exit.")
    
    steps = 0
    score = 0.0
    direction = 0  # UP
    
    # Animation loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Render current state
        viz.render(snake, food, obstacles, steps, score, direction)
        
        # Simulate movement (move snake up)
        steps += 1
        score += -1
        
        # Move snake head up
        head_x, head_y = snake[0]
        new_head = (head_x, head_y - 1)
        
        # Check if we hit the top
        if new_head[1] < 0:
            print("\nSnake hit the top! Resetting...")
            snake = [(4, 4), (4, 5), (4, 6)]
            steps = 0
            score = 0.0
        else:
            # Move snake
            snake.insert(0, new_head)
            snake.pop()
        
        # Wait a bit between frames
        pygame.time.wait(300)
    
    viz.close()
    print("Test complete!")