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

permutations_list = list(product((-0.5, 0.0, 0.5), repeat=3))
ACTIONS = [perm for perm in permutations_list]

# Variables
test_rewards = []

def distance(pose1, pose2):
    return math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)

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


def getState(x, y, theta, target_pos):
    dx = target_pos[0] - x
    dy = target_pos[1] - y
    angle = math.degrees(math.atan2(dy, dx)) - theta

    if angle < -180:
        angle += 360
    elif angle > 180:
        angle -= 360

    if abs(angle) <= 45:
        return 0  # ball in front of the robot
    elif abs(angle) >= 135:
        return 1  # ball behind the robot
    elif angle < -45:
        return 2  # ball to the left of the robot
    elif angle > 45 and angle < 135:
        return 3  # ball to the right of the robot

def rewardFunction(robot_pos, target_pos):
    x, y, theta = robot_pos
    current_distance_to_target = distance([x, y], target_pos)
    flag = True

    if not FIELD.collidepoint(x, y):
        reward = -10
        flag = False
    elif current_distance_to_target <= ROBOT_RADIUS:
        reward = 1
        flag = False
    else:
        reward = -1 / current_distance_to_target

    return reward, flag

Q = np.load("./Trained Files/policy.npy")
training_rewards = np.load("./Trained Files/rewards.npy")

# Testing
test_rewards = []
for episode in range(50):
    score = 0
    robot_pos = [random.randint(FIELD.left , FIELD.right), random.randint(FIELD.top, FIELD.bottom), random.randint(0, 359)]
    target_pos = [random.randint(FIELD.left , FIELD.right), random.randint(FIELD.top, FIELD.bottom)]
    steps = 1000
    for step in range(steps):
        current_state = getState(robot_pos[0], robot_pos[1], robot_pos[2], target_pos)
        action = np.argmax(Q[current_state])
        omega_0 = ACTIONS[action][0]
        omega_1 = ACTIONS[action][1]
        omega_2 = ACTIONS[action][2]
        robot_pos_prime = update_pose(robot_pos[0], robot_pos[1], robot_pos[2], omega_0, omega_1, omega_2, 0.1)
        reward, flag= rewardFunction(robot_pos_prime, target_pos)
        robot_pos = robot_pos_prime
        score += reward
        # if not flag:
        #     break
    test_rewards.append(score)

# Plotting
import matplotlib.pyplot as plt
plt.plot(training_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards over Episodes')
plt.show()

# Outputting the results
print("Average reward:", sum(test_rewards)/len(test_rewards))
print("Standard deviation:", np.std(test_rewards))
print("Testing rewards:", test_rewards)
print("Training rewards:", training_rewards)
print("Highest reward:", max(test_rewards))