import pygame
import random # Add missing import

# Constants
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Game initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()

# Snake variables
snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
snake_direction = (1, 0)

# Food variables
food = (GRID_WIDTH // 4, GRID_HEIGHT // 4)


def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (WIDTH, y))

# Game loop
running = True
while running:
    # --- Handle Events (Single Loop) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake_direction != (0, 1):
                snake_direction = (0, -1)
            elif event.key == pygame.K_DOWN and snake_direction != (0, -1):
                snake_direction = (0, 1)
            elif event.key == pygame.K_LEFT and snake_direction != (1, 0):
                snake_direction = (-1, 0)
            elif event.key == pygame.K_RIGHT and snake_direction != (-1, 0):
                snake_direction = (1, 0)

    new_head = ((snake[0][0] + snake_direction[0]) % GRID_WIDTH, (snake[0][1] + snake_direction[1]) % GRID_HEIGHT)
    snake.insert(0, new_head)

    if new_head == food:
        food = (random.randrange(0, GRID_WIDTH), random.randrange(0, GRID_HEIGHT))
    else:
        snake.pop()

    # Check for game over
    # --- End Handle Events ---

    # --- Game Logic (Single Block) ---
    new_head = ((snake[0][0] + snake_direction[0]) % GRID_WIDTH, (snake[0][1] + snake_direction[1]) % GRID_HEIGHT)
    snake.insert(0, new_head)

    if new_head == food:
        # Generate new food location, ensuring it's not inside the snake
        while True:
            food = (random.randrange(0, GRID_WIDTH), random.randrange(0, GRID_HEIGHT))
            if food not in snake:
                break
    else:
        snake.pop()

    # Check for game over (collision with self)
    if new_head in snake[1:]:
        running = False
    # --- End Game Logic ---

    # --- Drawing ---
    screen.fill(BLACK)
    draw_grid()
    # Fix indentation for drawing snake and food
    for segment in snake:
        pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.draw.rect(screen, WHITE, (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    pygame.display.flip()
    # --- End Drawing ---
    clock.tick(10)

pygame.quit()
