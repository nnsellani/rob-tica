"""
IRI - TP2 - Ex 1
By: Gonçalo Leão
"""

from controller import Robot, TouchSensor
from controllers.utils import cmd_vel

# Create the Robot instance.
robot: Robot = Robot()

timestep: int = int(robot.getBasicTimeStep())  # in ms

touch_sensor: TouchSensor = robot.getDevice('touch sensor')
touch_sensor.enable(timestep)

linear_vel: float = 0.2
while True:
    # Process the sensor
    if touch_sensor.getValue() == 1.0:
        print('bonk!')
        cmd_vel(robot, -0.1, 0)
        robot.step(1000)
        cmd_vel(robot, 0, 1)
        robot.step(1000)
    else:
        cmd_vel(robot, 0.1, 0)
        robot.step()
