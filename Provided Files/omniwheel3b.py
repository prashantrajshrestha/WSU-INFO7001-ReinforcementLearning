#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:22:41 2023

@author: 30045063
"""

import pygame
import random
import math

# Initialise Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FIELD = pygame.Rect(50, 50, WIDTH-100, HEIGHT-100)
ROBOT_RADIUS = 20
WHEEL_RADIUS = 5
TARGET_RADIUS = 10
FONT = pygame.font.SysFont("Arial", 24)

def new_episode(episode = -1):
    robot_pose = [random.randint(FIELD.left, FIELD.right), 
                  random.randint(FIELD.top, FIELD.bottom),
                  random.randint(0,359)]    
    target_pos = [random.randint(FIELD.left, FIELD.right), 
                  random.randint(FIELD.top, FIELD.bottom)]
    target_vel = [random.uniform(0.2, 0.6),
                  random.uniform(-0.6, 0.6)]
    
    return robot_pose, target_pos, target_vel, episode + 1, 0


def clip(value, min_val = -1, max_val = 1):
    return max(min(value, max_val), min_val)


def update_pose(x, y, theta, omega_0, omega_1, omega_2, step_size=1.0):
    omega_0 = clip(omega_0)
    omega_1 = clip(omega_1)
    omega_2 = clip(omega_2)
    
    R = 0.5
    # d = 1.0
    V_x = R * (omega_0 * math.cos(math.radians(60)) +
               omega_1 * math.cos(math.radians(180)) +
               omega_2 * math.cos(math.radians(300)))
    V_y = R * (omega_0 * math.sin(math.radians(60)) +
               omega_1 * math.sin(math.radians(180)) +
               omega_2 * math.sin(math.radians(300)))
    V_x_rotated = (V_x * math.cos(math.radians(-theta)) - 
                   V_y * math.sin(math.radians(-theta)))
    V_y_rotated = (V_x * math.sin(math.radians(-theta)) + 
                   V_y * math.cos(math.radians(-theta)))

    omega = omega_0 + omega_1 + omega_2
    x_prime = x + V_x_rotated * step_size
    y_prime = y + V_y_rotated * step_size

    theta_prime = theta + omega * step_size
    theta_prime = theta_prime % 360
    return x_prime, y_prime, theta_prime

# Start first episode
score = 0
running = True
omega_0, omega_1, omega_2 = 0, 0, 0

[x,y,theta], target_pos, target_vel, episode, step = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # Update robot
    omega_0 = random.uniform(-1, 1)
    omega_1 = random.uniform(-1, 1)
    omega_2 = random.uniform(-1, 1)


    x, y, theta = update_pose(x, y, theta, omega_0, omega_1, omega_2)
    step += 1
    score -= 0.01

    # Update target position
    target_pos[0] += target_vel[0]
    target_pos[1] += target_vel[1]

    # bounce off the sides
    if target_pos[0] <= 50 or target_pos[0] >= WIDTH - 50:
        target_vel[0] = -target_vel[0]
    if target_pos[1] <= 50 or target_pos[1] >= HEIGHT - 50:
        target_vel[1] = -target_vel[1]

    # Check for target, timeout, or out-of-bounds
    distance_to_target = math.sqrt((x - target_pos[0])**2 + (y - target_pos[1])**2)
    if distance_to_target <= ROBOT_RADIUS:
        score += 10
        [x,y,theta], target_pos, target_vel, episode, step = new_episode(episode)
    elif not FIELD.collidepoint(x, y):
        score -= 10
        [x,y,theta], target_pos, target_vel, episode, step = new_episode(episode)
    elif step > 1000:
        [x,y,theta], target_pos, target_vel, episode, step = new_episode(episode)
        
    # Draw everything
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (25, 25, 25), FIELD)

    pygame.draw.circle(screen, (200, 200, 200), (int(x), HEIGHT - int(y)), ROBOT_RADIUS)
    pygame.draw.circle(screen, (255, 165, 0), (int(target_pos[0]), HEIGHT - int(target_pos[1])), TARGET_RADIUS)
    
    # Draw Wheels
    for i, colour in zip([60, 180, 300], [(255, 0, 0), (255, 0, 255), (0, 0, 255)]):
        wheel_x = int(x + ROBOT_RADIUS * math.cos(math.radians(i + theta - 90)))
        wheel_y = HEIGHT - int(y - ROBOT_RADIUS * math.sin(math.radians(i + theta - 90)))
        pygame.draw.circle(screen, colour, (wheel_x, wheel_y), WHEEL_RADIUS)
    
    score_surface = FONT.render(f'Episode: {episode}  Step: {step}  Score: {score:.2f}', True, (255, 255, 255))
    screen.blit(score_surface, (WIDTH - 400, 10))
    
    pygame.display.flip()
    pygame.time.delay(50)
