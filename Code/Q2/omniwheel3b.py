#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:22:41 2023

@author: 30045063
"""

import pygame
import random
import math
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
EPSILON = 1 #exploration rate

# Creating the actions list increasing the speed to 1
permutations_list = list(product((-1, 0.0, 1), repeat=3))
ACTIONS = [perm for perm in permutations_list]

NUM_STATES = FIELD.width * FIELD.height
NUM_ACTIONS = len(ACTIONS)

reward_list = []

def new_episode(episode = -1, episode_reward = 0):

    global EPSILON 
    EPSILON = max(EPSILON *( 1 - 0.0095), 0.01)
    reward_list.append(episode_reward) 
    print(EPSILON)

    if episode < 500:
        max_distance = max(50 , episode )
    else:
        max_distance = 9999
    while True :
        robot_pose = [random.randint(FIELD.left , FIELD.right),
        random.randint(FIELD.top, FIELD.bottom), 0] # random.randint(0, 359)

        target_pos = [random.randint(FIELD.left , FIELD.right), random.randint(FIELD.top, FIELD.bottom)]
        target_vel = [random.uniform(0.2, 0.6), random.uniform(-0.6, 0.6)]
        d = distance(robot_pose , target_pos)

        if d <= max_distance:
            break
    
    return robot_pose, target_pos, target_vel, episode + 1, 0, 0


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
    return x_prime, y_prime, theta_prime, V_x_rotated, V_y_rotated

# Precompute rotation matrix
def create_rotation_matrix(theta):
    return np.array([[np.cos(-theta), -np.sin(-theta)],
                     [np.sin(-theta),  np.cos(-theta)]])

def getState(x, y, theta, target_pos, target_velocity):
    # Unpack the target position
    x_target, y_target = target_pos

    relative_position = np.array([x_target, y_target]) - np.array([x, y])

    rotation_matrix = create_rotation_matrix(theta)
    relative_position_rotated = np.dot(rotation_matrix, relative_position)

    x_rel_rotated = relative_position_rotated[0]
    y_rel_rotated = relative_position_rotated[1]

    if y_rel_rotated > 0 and abs(x_rel_rotated) < ROBOT_RADIUS:
        state_position = 0 # robot is directly front the target
    elif y_rel_rotated > 0 and x_rel_rotated < 0:
        state_position = 1 # robot is to the left of the target
    elif y_rel_rotated > 0 and x_rel_rotated >= 0:
        state_position = 2 # robot is to the right of the target
    else:
        state_position = 3

    # Calculate horizontal_velocity
    if target_velocity[0] >= 0.5:
        horizontal_velocity = 0
    elif -0.5 <= target_velocity[0] < 0.5:
        horizontal_velocity = 1
    elif target_velocity[0] < -0.5:
        horizontal_velocity = 2

    # Calculate vertical_velocity
    if target_velocity[1] >= 0.5:
        vertical_velocity = 0
    elif -0.5 <= target_velocity[1] < 0.5:
        vertical_velocity = 1
    elif target_velocity[1] < -0.5:
        vertical_velocity = 2

    # Combine the position and velocity states
    state = state_position * 9 + horizontal_velocity * 3 + vertical_velocity
    return state
        
    

def update_target_pose(target_pos, target_vel, step_size=0.1):
    target_x, target_y = target_pos
    velocity_x, velocity_y = target_vel
    target_x += velocity_x * step_size 
    target_y += velocity_y * step_size

    return [target_x, target_y], target_vel

def get_robot_velocity(omega_0, omega_1, omega_2, R=ROBOT_RADIUS, r=WHEEL_RADIUS):
    # Calculate the robot's velocity in its local coordinate system
    V_x = -r/3 * (omega_0 * np.sqrt(3) + omega_1 * np.sqrt(3))
    V_y = r/3 * (2*omega_0 - omega_1 - omega_2)
    # V_theta = -r/(3*R) * (omega_0 + omega_1 + omega_2)
    return [V_x, V_y]

def distance(pose1, pose2):
    return (abs(pose1[0] - pose2[0]) + abs(pose1[1]-pose2[1])) # using manhattan distance

def clip(value, min_val = -1, max_val = 1):
    return max(min(value, max_val), min_val)

# Initialise Q-table
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# Start first episode
score = 0
running = True
omega_0, omega_1, omega_2 = 0, 0, 0
V_x_rotated, V_y_rotated = 0, 0
distance_to_target = 10
current_distance_to_target = 10
current_velocity_towards_target = 0
reward = 0
episode_reward = 0

[x,y,theta], target_pos, target_vel, episode, step, episode_reward = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    ## Choose action
    if np.random.uniform() < EPSILON:
        action = np.random.randint(NUM_ACTIONS)
        state = getState(x, y, theta, target_pos, target_vel)
    else:
        state = getState(x, y, theta, target_pos, target_vel)
        action = np.argmax(Q[state])

    # Update robot
    omega_0 = ACTIONS[action][0]
    omega_1 = ACTIONS[action][1]
    omega_2 = ACTIONS[action][2]

    x, y, theta, V_x_rotated, V_y_rotated = update_pose(x, y, theta, omega_0, omega_1, omega_2)
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

    # Reward Function
    previous_distance_to_target = current_distance_to_target
    previous_velocity_towards_target = current_velocity_towards_target
    current_distance_to_target = distance([x, y], target_pos)

    # Calculate the vector pointing from the robot to the target
    target_vector = np.array([target_pos[0] - x, target_pos[1] - y])

    # Normalize the target vector to get the unit vector
    target_unit_vector = target_vector / np.linalg.norm(target_vector)

    # Calculate the dot product of the robot's velocity vector and the target unit vector
    current_velocity_towards_target = np.dot(np.array([V_x_rotated, V_y_rotated]), target_unit_vector)
        
    if step > 1000:
        # Start a new episode if the episode has timed out
        [x, y, theta], target_pos, target_vel, episode, step, episode_reward = new_episode(episode, episode_reward)
        reward = 0
    elif not FIELD.collidepoint(x, y):
        # Penalize the robot for going out of bounds
        score -= 1
        reward = -10
        [x, y, theta], target_pos, target_vel, episode, step, episode_reward = new_episode(episode, episode_reward)
    elif current_distance_to_target <= ROBOT_RADIUS:
        # Reward the robot for colliding with the target
        reward = 10
        score += 10
        [x, y, theta], target_pos, target_vel, episode, step, episode_reward = new_episode(episode, episode_reward)
    # elif current_distance_to_target == previous_distance_to_target:
    #     reward = 0.05  # Reward for staying still
    # elif current_distance_to_tcarget < previous_distance_to_target:
    #     reward = 0.1
    # #     score += 0.01
    elif current_velocity_towards_target > previous_velocity_towards_target:
        reward = 0.1
        score += 0.01
    # else:
    #     reward = -0.01
    #     score -= 0.01
    


    distance_to_target = current_distance_to_target
    episode_reward += reward

    
    # Update distance to target
    distance_to_target = current_distance_to_target

    # Update Q-table
    next_state = getState(x, y, theta, target_pos, target_vel)
    next_action = np.argmax(Q[next_state])
    Q[state, action] += ALPHA * (reward + GAMMA * Q[next_state, next_action] - Q[state, action])
    # state = next_state
    # action = next_action

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

np.save('policyQ2.npy', Q)
np.save('rewardsQ2.npy', reward_list)
