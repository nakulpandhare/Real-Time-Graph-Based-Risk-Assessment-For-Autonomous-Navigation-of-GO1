# robot_sim.py
import pygame
import random

# Pygame setup
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
radius = 20
speed = 5

def run_simulation(emergency_stop_flag):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Random Moving Circle")

    x, y = WIDTH // 2, HEIGHT // 2
    dx = random.choice([-1, 1]) * speed
    dy = random.choice([-1, 1]) * speed
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if emergency_stop_flag.value == 0:
            x += dx
            y += dy

            if x - radius <= 0 or x + radius >= WIDTH:
                dx = -dx + random.choice([-1, 0, 1])
            if y - radius <= 0 or y + radius >= HEIGHT:
                dy = -dy + random.choice([-1, 0, 1])

            pygame.draw.circle(screen, BLUE, (x, y), radius)
        else:
            pygame.draw.circle(screen, RED, (x, y), radius)

        pygame.display.flip()

    pygame.quit()
