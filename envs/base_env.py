import gymnasium
from gymnasium import spaces
import numpy as np
from numpy import random
import math
import random

import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Supervisor
from webots.controllers.utils import cmd_vel
from env.my_env import MyEnv


class DeterministicEnv(MyEnv):

    def __init__(self, supervisor: Supervisor):
        super(DeterministicEnv, self).__init__(supervisor)

    def step(self, action=None):
        linear_vel, angular_vel = self.distance_handler(1, self.lidar.getRangeImage())
        cmd_vel(self.supervisor, linear_vel, angular_vel)

        step_result = self.supervisor.step()

        self.previous_distance_from_goal = self.current_distance_from_goal
        self.current_distance_from_goal = self.get_distance_from_goal()

        observation = np.array(self.lidar.getRangeImage())
        reward = self.get_current_reward()
        done = (step_result == -1
                or self.current_distance_from_goal < 0.05
                or self.touch_sensor.getValue() == 1)

        if done:
            self.reset()

        # Execute action and return observation, reward, done, truncated, info
        return observation, reward, done, False, {}

    def distance_handler(self, direction: int, dist_values: [float]) -> (float, float):
        maxSpeed: float = 0.1
        distP: float = 10.0  # 10.0
        angleP: float = 7.0  # 7.0
        wallDist: float = 0.1

        # Find the angle of the ray that returned the minimum distance
        size: int = len(dist_values)
        min_index: int = 0
        if direction == -1:
            min_index = size - 1
        for i in range(size):
            idx: int = i
            if direction == -1:
                idx = size - 1 - i
            if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
                min_index = idx

        angle_increment: float = 2 * math.pi / (size - 1)
        angleMin: float = (size // 2 - min_index) * angle_increment
        distMin: float = dist_values[min_index]
        distFront: float = dist_values[size // 2]
        distSide: float = dist_values[size // 4] if (direction == 1) else dist_values[3 * size // 4]
        distBack: float = dist_values[0]

        # Prepare message for the robot's motors
        linear_vel: float
        angular_vel: float

        print("distMin", distMin)
        print("angleMin", angleMin * 180 / math.pi)

        # Decide the robot's behavior
        if math.isfinite(distMin):
            if distFront < 1.25 * wallDist and (distSide < 1.25 * wallDist or distBack < 1.25 * wallDist):
                print("UNBLOCK")
                angular_vel = direction * -1
            else:
                print("REGULAR")
                angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)
                print("angular_vel", angular_vel, " wall comp = ", direction * distP * (distMin - wallDist),
                      ", angle comp = ", angleP * (angleMin - direction * math.pi / 2))
            if distFront < wallDist:
                # TURN
                print("TURN")
                linear_vel = 0
            elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
                # SLOW
                print("SLOW")
                linear_vel = 0.5 * maxSpeed
            else:
                # CRUISE
                print("CRUISE")
                linear_vel = maxSpeed
        else:
            # WANDER
            print("WANDER")
            angular_vel = random.normal(loc=0.0, scale=1.0)
            print("angular_vel", angular_vel)
            linear_vel = maxSpeed

        return linear_vel, angular_vel


if __name__ == '__main__':
    supervisor = Supervisor()
    try:
        env = DeterministicEnv(supervisor)

        while True:
            env.step()
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()