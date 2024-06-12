import gymnasium
import numpy as np
import random
import time

import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Supervisor, Lidar, GPS, TouchSensor


def _my_activation_func(x):
    return ((1 / (1 + np.e ** -x)) - .5) * 2


class MyEnv(gymnasium.Env):
    N_OBSERVATIONS = 120
    WALL_DISTANCE_GOAL = 0.08
    MAX_WALL_DISTANCE = 0.125
    TIME_LIMIT = 60
    STEPS_PER_ACTION = 1

    INITIAL_AND_FINAL_STATES = [
        ((0.64, 0.83), (0.64, 0.11), -3.0),
        ((0.64, 0.26), (0.64, 0.68), -3.2),
        ((1.76, 0.27), (1.28, 0.78), 1.6),
        ((1.28, 0.61), (1.59, 0.27), 0),
        ((1.39, 1.21), (1.39, 1.76), 0.5),
        ((1.32, 1.62), (1.32, 1.36), -0.4),
        ((0.31, 1.26), (0.77, 1.42), 0),
        ((0.68, 1.55), (1.31, 1.41), -2.4)
    ]

    def __init__(self, supervisor: Supervisor):
        super(MyEnv, self).__init__()

        self.supervisor = supervisor

        self.start_time = None
        self.initial_pos = None
        self.final_pos = None
        self.current_distance_from_goal = None
        self.previous_distance_from_goal = None

        self.current_location = self.previous_location = (0, 0)

        self._init_robot()

        self.reset()

    def _init_robot(self):
        self.trans_field = self.supervisor.getFromDef('EPUCK').getField('translation')
        self.rotat_field = self.supervisor.getFromDef('EPUCK').getField('rotation')

        self.lidar: Lidar = self.supervisor.getDevice('lidar')
        self.gps: GPS = self.supervisor.getDevice('gps')

        timestep: int = int(self.supervisor.getBasicTimeStep())  # in ms

        self.lidar.enable(timestep)
        self.lidar.enablePointCloud()

        self.gps.enable(timestep)

        self.touch_sensor: TouchSensor = self.supervisor.getDevice('touch sensor')
        self.touch_sensor.enable(timestep)

    def reset(self, seed=None):
        self.start_time = time.time()

        states = list(random.choice(self.INITIAL_AND_FINAL_STATES))
        #random.shuffle(states)
        self.initial_pos, self.final_pos, angle = states

        self.current_distance_from_goal = self.previous_distance_from_goal = self.get_distance_from_goal()

        # set robot position
        self.trans_field.setSFVec3f([self.initial_pos[0], self.initial_pos[1], 0])
        self.rotat_field.setSFRotation([0, 0, 1, angle])

        observation = self.get_my_lidar_readings()
        return observation, {}

    def get_current_reward(self, wd_multiplier=.2, progress_multiplier=.03, movement_multiplier=.05):
        wall_distance_reward = (1 - (np.abs(np.min(self.get_my_lidar_readings()[28:33]) - self.WALL_DISTANCE_GOAL)) ** 1/10)
        progress_reward = (self.previous_distance_from_goal - self.current_distance_from_goal)
        #efficiency_reward = self.get_time_elapsed() * -1
        movement_reward = self.get_distanced_traveled() * movement_multiplier

        total_reward = (wall_distance_reward * wd_multiplier
                        + progress_reward * progress_multiplier
                        #+ efficiency_reward * time_multiplier
                        + movement_reward * movement_multiplier
                        )

        #total_reward = _my_activation_func(total_reward)

        return total_reward

    def get_distance_from_goal(self):
        gps_readings: [float] = self.gps.getValues()
        current_pos: (float, float) = (gps_readings[0], gps_readings[1])

        return np.sqrt((current_pos[0] - self.final_pos[0]) ** 2 + (current_pos[1] - self.final_pos[1]) ** 2)

    def get_distanced_traveled(self):
        return np.sqrt((self.current_location[0] - self.previous_location[0]) ** 2 +
                       (self.current_location[1] - self.previous_location[1]) ** 2)

    def get_my_lidar_readings(self):
        readings = np.array(self.lidar.getRangeImage())
        readings[readings > 3] = 3
        return readings

    def check_wall_collision(self, min_distance=.045):
        min_lidar_reading = np.min(self.get_my_lidar_readings())
        if min_lidar_reading < min_distance:
            return True
        return False

    def check_max_wall_distance_crossed(self):
        min_lidar_reading = np.min(self.get_my_lidar_readings())
        if min_lidar_reading > self.MAX_WALL_DISTANCE:
            return True
        return False

    def get_time_elapsed(self):
        return time.time() - self.start_time
