import math

theta = 0

x, y = 12, 12
target_pos = (10, 10)

def angleBetween(x, y, target_pos):
    return math.radians(math.atan2(target_pos[1] - y, target_pos[0] - x))

angleDiff = angleBetween(x, y, target_pos) - theta

if angleDiff > 45 and angleDiff < -45:
    return 0
else:
    return 1



## Q-learning Algorithm

# Initialize Q-table to zeros
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# Set hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Start first episode
score = 0
running = True
omega_0, omega_1, omega_2 = 0, 0, 0

[x, y, theta], target_pos, episode, step = new_episode()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # Choose action
    if np.random.uniform() < epsilon:
        # Explore
        action = np.random.randint(NUM_ACTIONS)
    else:
        # Exploit
        state = state(x, y, theta, target_pos)
        action = np.argmax(Q[state])

    # Update robot
    omega_0 = ACTIONS[action][0]
    omega_1 = ACTIONS[action][1]
    omega_2 = ACTIONS[action][2]

    x, y, theta = update_pose(x, y, theta, omega_0, omega_1, omega_2)
    step += 1
    score -= 0.01

    # Get reward
    distance_to_target = distance([x, y], target_pos)
    if distance_to_target < TARGET_RADIUS:
        reward = 1
        score += 1
        [x, y, theta], target_pos, episode, step = new_episode()
    else:
        reward = -0.1

    # Update Q-table
    next_state = state(x, y, theta, target_pos)
    next_action = np.argmax(Q[next_state])
    Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    state = next_state
    action = next_action

    # Update screen
    screen.fill(BLACK)
    draw_robot(screen, x, y, theta)
    draw_target(screen, target_pos)
    pygame.display.flip()

# Quit Pygame
pygame.quit()