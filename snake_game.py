#!/usr/bin/env python3
"""
Enhanced Snake Game

A modern implementation of the classic Snake game with additional features:
- Multiple difficulty levels
- Score tracking and high score saving
- Special food items with different effects
- Visual effects and animations
- Sound effects (if pygame.mixer is available)

Controls:
- Arrow keys: Control snake direction
- P: Pause game
- R: Restart game
- Q: Quit game
- 1-3: Change difficulty level

Created using FEI (Flying Dragon of Adaptability) AI assistant
"""

import pygame
import sys
import random
import time
import os
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
SNAKE_SPEED_EASY = 8
SNAKE_SPEED_MEDIUM = 12
SNAKE_SPEED_HARD = 16

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 100, 0)

# Direction enum
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

# Food types enum
class FoodType(Enum):
    REGULAR = 1    # Regular food, increases length by 1
    BONUS = 2      # Bonus food, increases score by more
    SPEED = 3      # Speed food, temporarily increases speed
    SLOW = 4       # Slow food, temporarily decreases speed
    SHRINK = 5     # Shrink food, decreases length

# Game states
class GameState(Enum):
    MENU = 0
    PLAYING = 1
    PAUSED = 2
    GAME_OVER = 3

class SnakeGame:
    """Enhanced Snake Game implementation"""
    
    def __init__(self):
        """Initialize the game"""
        # Set up the screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Enhanced Snake Game")
        
        # Set up the clock
        self.clock = pygame.time.Clock()
        
        # Set up fonts
        self.font_large = pygame.font.SysFont('Arial', 48)
        self.font_medium = pygame.font.SysFont('Arial', 36)
        self.font_small = pygame.font.SysFont('Arial', 24)
        
        # Set up sound effects if mixer is available
        self.sounds = {}
        try:
            pygame.mixer.init()
            self.sounds_enabled = True
            
            # Create sounds directory if it doesn't exist
            sounds_dir = Path("sounds")
            sounds_dir.mkdir(exist_ok=True)
            
            # We'll use placeholder sounds for now
            # In a real game, you'd want to include actual sound files
            self._create_placeholder_sounds()
            
        except pygame.error:
            self.sounds_enabled = False
            print("Sound mixer not available. Running without sound effects.")
        
        # Initialize game state
        self.state = GameState.MENU
        self.difficulty = 1  # 0: Easy, 1: Medium, 2: Hard
        self.speed_multiplier = 1.0
        self.score = 0
        self.high_score = self._load_high_score()
        
        # Initialize snake
        self.reset_game()
    
    def _create_placeholder_sounds(self):
        """Create placeholder sound effects"""
        # This is just for demonstration - in a real game, you'd load actual sound files
        sounds_dir = Path("sounds")
        
        # Create simple beep sounds of different frequencies for different actions
        for name, freq, duration in [
            ("eat", 440, 0.1),      # A4 note
            ("bonus", 660, 0.1),    # E5 note
            ("crash", 220, 0.3),    # A3 note
            ("shrink", 330, 0.1),   # E4 note
            ("speed", 550, 0.1),    # C#5 note
            ("slow", 294, 0.1),     # D4 note
        ]:
            sound_path = sounds_dir / f"{name}.wav"
            
            # Only create if it doesn't exist
            if not sound_path.exists():
                self._create_beep_sound(sound_path, freq, duration)
            
            # Load the sound
            try:
                self.sounds[name] = pygame.mixer.Sound(str(sound_path))
            except pygame.error:
                print(f"Could not load sound: {sound_path}")
    
    def _create_beep_sound(self, path: Path, frequency: int, duration: float):
        """Create a simple beep sound file"""
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Generate a simple sine wave
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(frequency * 2 * np.pi * t) * 0.5
            
            # Convert to 16-bit PCM
            wave = (wave * 32767).astype(np.int16)
            
            # Save as WAV file
            wavfile.write(path, sample_rate, wave)
            
        except ImportError:
            print("Could not create sound effects. numpy and scipy are required.")
    
    def _load_high_score(self) -> int:
        """Load high score from file"""
        try:
            high_score_file = Path.home() / ".snake_high_score"
            if high_score_file.exists():
                with open(high_score_file, 'r') as f:
                    return int(f.read().strip())
        except (IOError, ValueError):
            pass
        return 0
    
    def _save_high_score(self):
        """Save high score to file"""
        try:
            high_score_file = Path.home() / ".snake_high_score"
            with open(high_score_file, 'w') as f:
                f.write(str(self.high_score))
        except IOError:
            print("Could not save high score.")
    
    def reset_game(self):
        """Reset the game state"""
        # Initialize snake in the middle of the screen
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.snake_direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        
        # Set speed based on difficulty
        if self.difficulty == 0:  # Easy
            self.speed = SNAKE_SPEED_EASY
        elif self.difficulty == 1:  # Medium
            self.speed = SNAKE_SPEED_MEDIUM
        else:  # Hard
            self.speed = SNAKE_SPEED_HARD
        
        # Reset score and effects
        self.score = 0
        self.speed_multiplier = 1.0
        self.effect_end_time = 0
        self.current_effect = None
        
        # Create initial food
        self.foods = []
        self._spawn_food()
    
    def _spawn_food(self):
        """Spawn a new food item"""
        # Determine food type based on probabilities
        food_type_roll = random.random()
        
        if food_type_roll < 0.7:
            food_type = FoodType.REGULAR
        elif food_type_roll < 0.8:
            food_type = FoodType.BONUS
        elif food_type_roll < 0.85:
            food_type = FoodType.SPEED
        elif food_type_roll < 0.9:
            food_type = FoodType.SLOW
        else:
            food_type = FoodType.SHRINK
        
        # Find a position that's not occupied by the snake
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in self.snake and pos not in [f[0] for f in self.foods]:
                break
        
        # Add the food to the list
        self.foods.append((pos, food_type))
    
    def _get_food_color(self, food_type: FoodType) -> Tuple[int, int, int]:
        """Get the color for a food type"""
        if food_type == FoodType.REGULAR:
            return RED
        elif food_type == FoodType.BONUS:
            return YELLOW
        elif food_type == FoodType.SPEED:
            return BLUE
        elif food_type == FoodType.SLOW:
            return PURPLE
        elif food_type == FoodType.SHRINK:
            return GRAY
        return RED  # Default
    
    def _apply_food_effect(self, food_type: FoodType):
        """Apply the effect of eating a food item"""
        if food_type == FoodType.REGULAR:
            # Regular food: increase length by 1, add 1 point
            self.score += 1
            # The snake will grow on the next move
            if self.sounds_enabled:
                self.sounds["eat"].play()
        
        elif food_type == FoodType.BONUS:
            # Bonus food: add 5 points
            self.score += 5
            if self.sounds_enabled:
                self.sounds["bonus"].play()
        
        elif food_type == FoodType.SPEED:
            # Speed food: temporarily increase speed by 50%
            self.speed_multiplier = 1.5
            self.effect_end_time = time.time() + 5  # 5 seconds
            self.current_effect = "SPEED"
            if self.sounds_enabled:
                self.sounds["speed"].play()
        
        elif food_type == FoodType.SLOW:
            # Slow food: temporarily decrease speed by 50%
            self.speed_multiplier = 0.5
            self.effect_end_time = time.time() + 5  # 5 seconds
            self.current_effect = "SLOW"
            if self.sounds_enabled:
                self.sounds["slow"].play()
        
        elif food_type == FoodType.SHRINK:
            # Shrink food: decrease length by 3 (minimum 1)
            if len(self.snake) > 3:
                self.snake = self.snake[:-3]
            self.score += 2
            if self.sounds_enabled:
                self.sounds["shrink"].play()
    
    def _check_collisions(self) -> bool:
        """Check for collisions with walls or self"""
        head = self.snake[0]
        
        # Check wall collision
        if (head[0] < 0 or head[0] >= GRID_WIDTH or
            head[1] < 0 or head[1] >= GRID_HEIGHT):
            return True
        
        # Check self collision (skip the head)
        if head in self.snake[1:]:
            return True
        
        return False
    
    def _check_food_collision(self) -> Optional[Tuple[Tuple[int, int], FoodType]]:
        """Check if the snake has eaten a food item"""
        head = self.snake[0]
        
        for food in self.foods:
            if food[0] == head:
                return food
        
        return None
    
    def _update_snake(self):
        """Update the snake's position"""
        # Check if any effect has expired
        if self.current_effect and time.time() > self.effect_end_time:
            self.speed_multiplier = 1.0
            self.current_effect = None
        
        # Update direction
        self.snake_direction = self.next_direction
        
        # Calculate new head position
        head = self.snake[0]
        direction = self.snake_direction.value
        new_head = (head[0] + direction[0], head[1] + direction[1])
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check for food collision
        food = self._check_food_collision()
        if food:
            # Apply food effect
            self._apply_food_effect(food[1])
            
            # Remove the food
            self.foods.remove(food)
            
            # Spawn a new food
            self._spawn_food()
            
            # For regular food, we keep the tail (snake grows)
            # For other food types, we remove the tail (snake stays same length)
            if food[1] != FoodType.REGULAR:
                self.snake.pop()
        else:
            # No food eaten, remove the tail
            self.snake.pop()
        
        # Check for collisions
        if self._check_collisions():
            if self.sounds_enabled:
                self.sounds["crash"].play()
            self.state = GameState.GAME_OVER
            
            # Update high score
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
    
    def _draw_snake(self):
        """Draw the snake on the screen"""
        # Draw snake body
        for i, segment in enumerate(self.snake):
            # Calculate position
            x = segment[0] * GRID_SIZE
            y = segment[1] * GRID_SIZE
            
            # Draw segment with gradient color (darker toward tail)
            if i == 0:  # Head
                color = GREEN
            else:  # Body
                # Create a gradient from green to dark green
                fade = min(1.0, i / len(self.snake))
                r = int(GREEN[0] * (1 - fade) + DARK_GREEN[0] * fade)
                g = int(GREEN[1] * (1 - fade) + DARK_GREEN[1] * fade)
                b = int(GREEN[2] * (1 - fade) + DARK_GREEN[2] * fade)
                color = (r, g, b)
            
            # Draw the segment
            pygame.draw.rect(self.screen, color, (x, y, GRID_SIZE, GRID_SIZE))
            
            # Draw a smaller rectangle inside for a 3D effect
            pygame.draw.rect(self.screen, (r+20, g+20, b+20) if i > 0 else (50, 255, 50), 
                            (x+4, y+4, GRID_SIZE-8, GRID_SIZE-8))
    
    def _draw_food(self):
        """Draw food items on the screen"""
        for pos, food_type in self.foods:
            # Calculate position
            x = pos[0] * GRID_SIZE
            y = pos[1] * GRID_SIZE
            
            # Get color based on food type
            color = self._get_food_color(food_type)
            
            # Draw the food
            pygame.draw.rect(self.screen, color, (x, y, GRID_SIZE, GRID_SIZE))
            
            # Add a visual indicator for special foods
            if food_type != FoodType.REGULAR:
                # Draw a symbol or pattern based on food type
                if food_type == FoodType.BONUS:
                    # Draw a star-like shape
                    pygame.draw.polygon(self.screen, WHITE, [
                        (x + GRID_SIZE//2, y + 4),
                        (x + GRID_SIZE - 4, y + GRID_SIZE//2),
                        (x + GRID_SIZE//2, y + GRID_SIZE - 4),
                        (x + 4, y + GRID_SIZE//2)
                    ])
                elif food_type == FoodType.SPEED:
                    # Draw an arrow
                    pygame.draw.polygon(self.screen, WHITE, [
                        (x + 4, y + GRID_SIZE//2),
                        (x + GRID_SIZE//2, y + 4),
                        (x + GRID_SIZE//2, y + GRID_SIZE - 4)
                    ])
                elif food_type == FoodType.SLOW:
                    # Draw a reverse arrow
                    pygame.draw.polygon(self.screen, WHITE, [
                        (x + GRID_SIZE - 4, y + GRID_SIZE//2),
                        (x + GRID_SIZE//2, y + 4),
                        (x + GRID_SIZE//2, y + GRID_SIZE - 4)
                    ])
                elif food_type == FoodType.SHRINK:
                    # Draw an X
                    pygame.draw.line(self.screen, WHITE, (x + 4, y + 4), 
                                    (x + GRID_SIZE - 4, y + GRID_SIZE - 4), 2)
                    pygame.draw.line(self.screen, WHITE, (x + 4, y + GRID_SIZE - 4), 
                                    (x + GRID_SIZE - 4, y + 4), 2)
    
    def _draw_grid(self):
        """Draw a grid on the screen"""
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (SCREEN_WIDTH, y))
    
    def _draw_score(self):
        """Draw the score on the screen"""
        score_text = self.font_small.render(f"Score: {self.score}", True, WHITE)
        high_score_text = self.font_small.render(f"High Score: {self.high_score}", True, WHITE)
        
        # Draw difficulty
        difficulty_names = ["Easy", "Medium", "Hard"]
        difficulty_text = self.font_small.render(f"Difficulty: {difficulty_names[self.difficulty]}", True, WHITE)
        
        # Draw current effect if any
        if self.current_effect:
            effect_text = self.font_small.render(f"Effect: {self.current_effect}", True, YELLOW)
            self.screen.blit(effect_text, (10, 70))
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (10, 40))
        self.screen.blit(difficulty_text, (SCREEN_WIDTH - 200, 10))
    
    def _draw_menu(self):
        """Draw the main menu"""
        # Draw title
        title_text = self.font_large.render("Enhanced Snake Game", True, GREEN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Draw instructions
        instructions = [
            "Use Arrow Keys to control the snake",
            "P: Pause game",
            "R: Restart game",
            "Q: Quit game",
            "1-3: Change difficulty level",
            "",
            "Food Types:",
            "Red: Regular food (grow)",
            "Yellow: Bonus food (extra points)",
            "Blue: Speed boost",
            "Purple: Slow down",
            "Gray: Shrink snake"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_small.render(line, True, WHITE)
            self.screen.blit(text, (SCREEN_WIDTH//2 - 150, 180 + i*30))
        
        # Draw "Press any key to start" message
        start_text = self.font_medium.render("Press SPACE to Start", True, YELLOW)
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH//2, 500))
        
        # Make it blink
        if int(time.time() * 2) % 2 == 0:
            self.screen.blit(start_text, start_rect)
    
    def _draw_game_over(self):
        """Draw the game over screen"""
        # Draw "Game Over" text
        game_over_text = self.font_large.render("Game Over", True, RED)
        game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, 200))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Draw score
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, 280))
        self.screen.blit(score_text, score_rect)
        
        # Draw high score
        high_score_text = self.font_medium.render(f"High Score: {self.high_score}", True, WHITE)
        high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH//2, 330))
        self.screen.blit(high_score_text, high_score_rect)
        
        # Draw "Press R to restart" message
        restart_text = self.font_medium.render("Press R to Restart", True, GREEN)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, 400))
        self.screen.blit(restart_text, restart_rect)
        
        # Draw "Press Q to quit" message
        quit_text = self.font_medium.render("Press Q to Quit", True, YELLOW)
        quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH//2, 450))
        self.screen.blit(quit_text, quit_rect)
    
    def _draw_pause(self):
        """Draw the pause screen"""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Draw "Paused" text
        paused_text = self.font_large.render("Paused", True, WHITE)
        paused_rect = paused_text.get_rect(center=(SCREEN_WIDTH//2, 200))
        self.screen.blit(paused_text, paused_rect)
        
        # Draw "Press P to resume" message
        resume_text = self.font_medium.render("Press P to Resume", True, GREEN)
        resume_rect = resume_text.get_rect(center=(SCREEN_WIDTH//2, 300))
        self.screen.blit(resume_text, resume_rect)
        
        # Draw "Press R to restart" message
        restart_text = self.font_medium.render("Press R to Restart", True, YELLOW)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, 350))
        self.screen.blit(restart_text, restart_rect)
        
        # Draw "Press Q to quit" message
        quit_text = self.font_medium.render("Press Q to Quit", True, RED)
        quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH//2, 400))
        self.screen.blit(quit_text, quit_rect)
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Global key handlers
                if event.key == pygame.K_q:
                    return False
                
                # Menu state
                if self.state == GameState.MENU:
                    if event.key == pygame.K_SPACE:
                        self.state = GameState.PLAYING
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                        self.difficulty = event.key - pygame.K_1
                        self.reset_game()
                
                # Playing state
                elif self.state == GameState.PLAYING:
                    if event.key == pygame.K_UP and self.snake_direction != Direction.DOWN:
                        self.next_direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.snake_direction != Direction.UP:
                        self.next_direction = Direction.DOWN
                    elif event.key == pygame.K_LEFT and self.snake_direction != Direction.RIGHT:
                        self.next_direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT and self.snake_direction != Direction.LEFT:
                        self.next_direction = Direction.RIGHT
                    elif event.key == pygame.K_p:
                        self.state = GameState.PAUSED
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                        self.difficulty = event.key - pygame.K_1
                        self.reset_game()
                
                # Paused state
                elif self.state == GameState.PAUSED:
                    if event.key == pygame.K_p:
                        self.state = GameState.PLAYING
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        self.state = GameState.PLAYING
                
                # Game over state
                elif self.state == GameState.GAME_OVER:
                    if event.key == pygame.K_r:
                        self.reset_game()
                        self.state = GameState.PLAYING
        
        return True
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            running = self._handle_events()
            
            # Clear the screen
            self.screen.fill(BLACK)
            
            # Draw grid
            self._draw_grid()
            
            # Update and draw based on game state
            if self.state == GameState.MENU:
                self._draw_menu()
            
            elif self.state == GameState.PLAYING:
                # Update snake at the appropriate speed
                current_speed = self.speed * self.speed_multiplier
                if pygame.time.get_ticks() % (1000 // int(current_speed)) < 20:
                    self._update_snake()
                
                # Draw game elements
                self._draw_snake()
                self._draw_food()
                self._draw_score()
            
            elif self.state == GameState.PAUSED:
                # Draw game elements
                self._draw_snake()
                self._draw_food()
                self._draw_score()
                self._draw_pause()
            
            elif self.state == GameState.GAME_OVER:
                # Draw game elements
                self._draw_snake()
                self._draw_food()
                self._draw_score()
                self._draw_game_over()
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(60)
        
        # Clean up
        pygame.quit()


if __name__ == "__main__":
    # Create and run the game
    game = SnakeGame()
    game.run()
