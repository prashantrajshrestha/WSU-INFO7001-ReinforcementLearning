#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:22:41 2023

@author: 30045063
"""

import pygame
import random
import math
import pandas as pd
import numpy as np
from itertools import permutations, product

# Initialise Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FIELD = pygame.Rect(50, 50, WIDTH-100, HEIGHT-100)
ROBOT_RADIUS = 20
WHEEL_RADIUS = 5
TARGET_RADIUS = 10
FONT = pygame.font.SysFont("Arial", 24)

ALPHA = 0.1 #learning rate
GAMMA = 0.9 #discount factor
EPSILON = 1.00 #exploration rate

# Creating the actions list
permutations_list = list(product((-1, 0.0, 1), repeat=3))
ACTIONS = [perm for perm in permutations_list]

NUM_STATES = 15
NUM_ACTIONS = len(ACTIONS)

def distance(pose1, pose2):
    return math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)

def clip(value, min_val = -1, max_val = 1):
    return max(min(value, max_val), min_val)

def new_episode(episode = -1):
    
    robot_pose = [random.randint(FIELD.left , FIELD.right),
    random.randint(FIELD.top, FIELD.bottom), 0] # random.randint(0, 359)

    target_pos = [random.randint(FIELD.left , FIELD.right), random.randint(FIELD.top, FIELD.bottom)]

    return robot_pose, target_pos, episode + 1, 0

def getState(x, y, theta, target_pos):
    dx = target_pos[0] - x
    dy = target_pos[1] - y
    angle = math.degrees(math.atan2(dy, dx)) - theta

    # print(angle)
    if angle < -180:
        angle += 360
    elif angle > 180:
        angle -= 360

    if abs(angle) <= 45:
        state = 0  # ball in front of the robot
    elif abs(angle) >= 135:
        state = 1  # ball behind the robot
    elif angle < -45:
        state = 2  # ball to the left of the robot
    elif angle > 45 and angle < 135:
        state = 3  # ball to the right of the robot
    
    return state

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
    V_x_rotated = (V_x * math.cos(math.radians(theta)) - 
                   V_y * math.sin(math.radians(theta)))
    V_y_rotated = (V_x * math.sin(math.radians(theta)) + 
                   V_y * math.cos(math.radians(theta)))

    omega = omega_0 + omega_1 + omega_2
    x_prime = x + V_x_rotated * step_size
    y_prime = y + V_y_rotated * step_size

    theta_prime = theta + omega * step_size
    theta_prime = theta_prime % 360
    return x_prime, y_prime, theta_prime

# Initialise Q-table
Q = np.load("./Trained Files - Q1/policy.npy")


# Start first episode
score = 0
running = True

[x,y,theta], target_pos, episode, step = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

episode_list=[]
step_list=[]
robot_pos_list=[]
target_pos_list=[]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    ## Choose action
    state = getState(x, y, theta, target_pos)
    action = np.argmax(Q[state])

    # Update robot
    omega_0 = ACTIONS[action][0]
    omega_1 = ACTIONS[action][1]
    omega_2 = ACTIONS[action][2]

    int_robot_pos=[x, y, theta]
    int_target_pos = target_pos
    x, y, theta = update_pose(x, y, theta, omega_0, omega_1, omega_2)
    step += 1
    score -= 0.01
    
    # Reward function and checking for target, timeout, or out-of-bounds
    current_distance_to_target = distance([x, y], target_pos)


    if step > 10000:
        # Start a new episode if the episode has timed out
        [x, y, theta], target_pos, episode, step= new_episode(episode-1)

    elif not FIELD.collidepoint(x, y):
        # Penalize the robot for going out of bounds
        [x, y, theta], target_pos, episode, step = new_episode(episode-1)
        score -= 1

    elif current_distance_to_target < TARGET_RADIUS:
        episode_list.append(episode)
        step_list.append(step)
        robot_pos_list.append(int_robot_pos)
        target_pos_list.append(int_target_pos)
        [x, y, theta], target_pos, episode, step= new_episode(episode)
        score += 10
        
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
    # pygame.time.delay(50)

data = {
    'episode': episode_list,
    'robot_pos':robot_pos_list,
    'target_pos':target_pos_list,
    'step': step_list
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the CSV file path
csv_file_path = 'data.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
