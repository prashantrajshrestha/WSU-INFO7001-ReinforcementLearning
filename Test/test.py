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