
import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Robot, TouchSensor
from webots.controllers.utils import cmd_vel


# Create the Robot instance.
robot: Robot = Robot()

timestep: int = int(robot.getBasicTimeStep())  # in ms

touch_sensor: TouchSensor = robot.getDevice('touch sensor')
touch_sensor.enable(timestep)

cmd_vel(robot, .1, -.5)
while True:
    robot.step()