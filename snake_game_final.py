import pygame
import random

# Constants
WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

# Snake
snake_speed = 15
snake_x = WIDTH // 2
snake_y = HEIGHT // 2
snake_body = [[snake_x, snake_y]]
snake_dx = BLOCK_SIZE
snake_dy = 0

# Draw Snake
def draw_snake(snake_body):
    for segment in snake_body:
        pygame.draw.rect(screen, GREEN, [segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE])

# Main game loop
game_over = False
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and snake_dx != BLOCK_SIZE:
                snake_dx = -BLOCK_SIZE
                snake_dy = 0
            elif event.key == pygame.K_RIGHT and snake_dx != -BLOCK_SIZE:
                snake_dx = BLOCK_SIZE
                snake_dy = 0
            elif event.key == pygame.K_UP and snake_dy != BLOCK_SIZE:
                snake_dx = 0
                snake_dy = -BLOCK_SIZE
            elif event.key == pygame.K_DOWN and snake_dy != -BLOCK_SIZE:
                snake_dx = 0
                snake_dy = BLOCK_SIZE

    pygame.display.update()
    clock.tick(snake_speed)

pygame.quit()
quit()