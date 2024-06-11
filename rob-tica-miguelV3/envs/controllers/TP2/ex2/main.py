"""
IRI - TP2 - Ex 2
By: Gonçalo Leão
"""

from controller import Robot, DistanceSensor
from controllers.utils import cmd_vel

# Create the Robot instance.
robot: Robot = Robot()

timestep: int = int(robot.getBasicTimeStep())  # in ms

distance_sensors: [DistanceSensor] = []
ds_names: [str] = ['ps0', 'ps1', 'ps6', 'ps7']
for ds_name in ds_names:
    ds: DistanceSensor = robot.getDevice(ds_name)
    ds.enable(timestep)
    distance_sensors.append(ds)

cam = robot.getDevice('camera')
cam.enable(timestep)

min_ds_value: float = 80.0
while robot.step() != -1:
    max_ds_reading: float = max([ds.getValue() for ds in distance_sensors])
    print([ds.getValue() for ds in distance_sensors])
    if max_ds_reading > min_ds_value:
        cmd_vel(robot, 0, 1)
    else:
        cmd_vel(robot, 0.1, 0)
