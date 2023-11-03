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

permutations_list = list(product((-1, 0.0, 1), repeat=3))
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
        state_position = 0
    elif y_rel_rotated > 0 and x_rel_rotated < 0:
        state_position = 1
    elif y_rel_rotated > 0 and x_rel_rotated >= 0:
        state_position = 2
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
    state = state_position * 4 + horizontal_velocity * 3 + vertical_velocity
    return state
        

def rewardFunction(robot_pos, target_pos, prev_distance_to_target = 1000):
    x, y, theta = robot_pos
    current_distance_to_target = distance([x, y], target_pos)

    if not FIELD.collidepoint(x, y):
        reward = -10
    elif current_distance_to_target <= ROBOT_RADIUS:
        reward = 1
    elif current_distance_to_target < prev_distance_to_target:
        reward = 0.1
    else:
        reward = -0.01

    return reward, current_distance_to_target

Q = np.load("./Trained Files/policyQ2.npy")
training_rewards = np.load("./Trained Files/rewardsQ2.npy")

# Testing
test_rewards = []
previous_distance_to_target = 1000
for episode in range(50):
    score = 0
    robot_pos = [random.randint(FIELD.left , FIELD.right), random.randint(FIELD.top, FIELD.bottom), random.randint(0, 359)]
    target_pos = [random.randint(FIELD.left , FIELD.right), random.randint(FIELD.top, FIELD.bottom)]
    target_vel = [random.randint(-1, 1), random.randint(-1, 1)]
    steps = 1000
    for step in range(steps):
        current_state = getState(robot_pos[0], robot_pos[1], robot_pos[2], target_pos, target_vel)
        action = np.argmax(Q[current_state])
        omega_0 = ACTIONS[action][0]
        omega_1 = ACTIONS[action][1]
        omega_2 = ACTIONS[action][2]
        robot_pos_prime = update_pose(robot_pos[0], robot_pos[1], robot_pos[2], omega_0, omega_1, omega_2, 0.1)
        reward, current_distance_to_target= rewardFunction(robot_pos_prime, target_pos, previous_distance_to_target)
        previous_distance_to_target = current_distance_to_target
        robot_pos = robot_pos_prime
        score += reward
        target_pos[0] += target_vel[0]
        target_pos[1] += target_vel[1]
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