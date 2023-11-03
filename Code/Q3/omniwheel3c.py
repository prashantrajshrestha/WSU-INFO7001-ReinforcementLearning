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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

reward_list = []

def new_episode(episode = -1):
    if episode < 60:
        max_distance = max(50 , episode )
    elif episode < 1000:
        max_distance = 9999
    else:
        running= False

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

def update_target_pose(target_pos, target_vel, step_size=0.1):
    target_x, target_y = target_pos
    velocity_x, velocity_y = target_vel
    target_x += velocity_x * step_size 
    target_y += velocity_y * step_size

    return [target_x, target_y], target_vel

def distance(pose1, pose2):
    return math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)

def update_pose(x, y, theta, target_pos, omega_0, omega_1, omega_2, step_size=1.0):
    proportional_gain = 0.3

    # Calculate error in position
    target_x, target_y = target_pos
    error_x = target_x - x
    error_y = target_y - y

    # Calculate control values based on error
    omega_0 += error_x * proportional_gain
    omega_1 += error_y * proportional_gain

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

def actionMatrix(action):
    control_values = [0.0, 0.0, 0.0]
    # Mapping actions to control values
    if action == 0:
        control_values = [1.0, 0.0, 0.0]  # up
    elif action == 1:
        control_values = [-1.0, 0.0, 0.0]  # down
    elif action == 2:
        control_values = [0.0, 1.0, 0.0]  # right
    elif action == 3:
        control_values = [0.0, -1.0, 0.0]  # left
    elif action == 4:
        control_values = [1.0, 1.0, 0.0] # up-right
    elif action == 5:
        control_values = [1.0, -1.0, 0.0] # up-left
    elif action == 6:
        control_values = [-1.0, 1.0, 0.0] # down-right
    elif action == 7:
        control_values = [-1.0, -1.0, 0.0] # down-left
    else:
        control_values = [0.0, 0.0, 0.0]  # error

    return control_values

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=-1)

        return x

# Creating the policy network
input_size = 2
hidden_size = 256
hidden_size2 = 128
actions = [0, 1, 2, 3, 4, 5, 6, 7]
output_size = len(actions)
policy = Policy(input_size, hidden_size, hidden_size2, output_size)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

# Function to train the policy network
def train_policy(state, action, reward_list):
    # Convert state to tensor
    state = state.float().unsqueeze(0)

    # Get action probabilities from policy network
    probs = policy(state)

    # Calculate log probability of the action taken
    log_prob = torch.log(probs.squeeze(0)[action])

    # Convert reward_list to tensor
    rewards = torch.tensor(reward_list)

    # Calculate loss
    loss = -torch.mean(log_prob * rewards)

    # Backpropagate loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()



# Start first episode
score = 0
running = True
omega_0, omega_1, omega_2 = 0, 0, 0

[x,y,theta], target_pos, target_vel, episode, step, episode_reward = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

episode = 0
reward_list = []
episode_reward = 0

while episode < 1000:
    [x,y,theta], target_pos, target_vel, episode, step, episode_reward = new_episode(episode)
    reward_list.append(episode_reward)
    episode_reward = 0
    step = 0
    distance_to_target = 10
    previous_distance_to_target = 10

    while step < 1000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False      
        
        if not running:
            break

        # RL Alrogithm
        state = torch.FloatTensor([x, y])
        action = policy(state)
        action_distribution = torch.distributions.Categorical(action)
        action = action_distribution.sample()

        # Perform robot action
        if step == 0:
            print(action.item())

        omega_0, omega_1, omega_2 = actionMatrix(action)

        # Update robot pose
        x_prime, y_prime, theta_prime = update_pose(x, y, theta, target_pos, omega_0, omega_1, omega_2)       

        previous_distance_to_target = distance_to_target
        distance_to_target = distance([x_prime, y_prime], target_pos)

        # Calculate reward based on distance to target
        if not FIELD.collidepoint(x, y):
            reward = -10
            score -= 10
            train_policy(state, action, reward_list)
            episode_reward += reward
            break
        elif distance_to_target < previous_distance_to_target:
            reward += 0.1
            score += 0.01
        elif distance_to_target <= ROBOT_RADIUS:
            reward = 10
            score += 10
            train_policy(state, action, reward_list)
            episode_reward += reward
            break
        else:
            reward = -10
            score -= 0.01

        # Train policy with the selected action and calculated reward
        train_policy(state, action, reward_list)
        episode_reward += reward

        # Update robot position
        x, y, theta = x_prime, y_prime, theta_prime

        # Update target position
        target_pos = update_target_pose(target_pos, target_vel)[0]

        # bounce off the sides
        if target_pos[0] <= 50 or target_pos[0] >= WIDTH - 50:
            target_vel[0] = -target_vel[0]
        if target_pos[1] <= 50 or target_pos[1] >= HEIGHT - 50:
            target_vel[1] = -target_vel[1]        

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
        # pygame.time.delay(5)

        step += 1

torch.save(policy.state_dict(), 'policy.pth')
np.save('rewardsQ3.npy', reward_list)
