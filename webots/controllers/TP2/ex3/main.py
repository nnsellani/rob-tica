"""
IRI - TP2 - Ex 3
By: Gonçalo Leão
"""

import math

from controller import Robot, GPS, Compass
from controllers.utils import cmd_vel

# Create the Robot instance.
robot: Robot = Robot()

timestep: int = int(robot.getBasicTimeStep())

gps: GPS = robot.getDevice('gps')
gps.enable(timestep)
compass: Compass = robot.getDevice('compass')
compass.enable(timestep)

POS_EPSILON: float = 1e-02
THETA_EPSILON: float = 1e-02
square_side_length: float = 0.25
while robot.step() != -1:
    position: [float] = gps.getValues()
    theta = math.atan2(compass.getValues()[0], compass.getValues()[1])
    # Bottom to right corner
    if abs(position[0] - square_side_length) < POS_EPSILON and abs(position[1]) < POS_EPSILON and abs(theta - math.pi/2) > THETA_EPSILON:
        print('Bottom to right corner')
        cmd_vel(robot, 0, 1)
    # Right to top corner
    elif abs(position[0] - square_side_length) < POS_EPSILON and abs(position[1] - square_side_length) < POS_EPSILON and abs(theta - math.pi) > THETA_EPSILON:
        print('Right to top corner')
        cmd_vel(robot, 0, 1)
    # Top to left corner
    elif abs(position[0]) < POS_EPSILON and abs(position[1] - square_side_length) < POS_EPSILON and abs(theta + math.pi/2) > THETA_EPSILON:
        print('Top to left corner')
        cmd_vel(robot, 0, 1)
    # Left to bottom corner
    elif abs(position[0]) < POS_EPSILON and abs(position[1]) < POS_EPSILON and abs(theta) > THETA_EPSILON:
        print('Left to bottom corner')
        cmd_vel(robot, 0, 1)
    else:
        cmd_vel(robot, 0.1, 0)