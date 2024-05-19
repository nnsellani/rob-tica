import gymnasium
import numpy as np
import random
import time

import sys
import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Supervisor, Lidar, GPS, TouchSensor

class MyEnv(gymnasium.Env):

    N_ACTION_SPACES = 5
    N_OBSERVATIONS = 180
    TIME_PENALTY_FACTOR = 1
    WALL_DISTANCE_GOAL = 0.2
    TIME_LIMIT = 60
    STEPS_PER_ACTION = 1

    INITIAL_AND_FINAL_STATES = [
        ((0.64, 0.11), (0.64, 0.83)),
        ((0.64, 0.26), (0.64, 0.68)),
        ((1.76, 0.27), (1.28, 0.78)),
        ((1.59, 0.27), (1.28, 0.61)),
        ((1.39, 1.21), (1.39, 1.76)),
        ((1.32, 1.62), (1.32, 1.36)),
        ((0.77, 1.42), (0.31, 1.26)),
        ((0.68, 1.55), (1.31, 1.41))
    ]

    def __init__(self, supervisor: Supervisor):
        super(MyEnv, self).__init__()

        self.supervisor = supervisor

        self.start_time = None
        self.initial_pos = None
        self.final_pos = None
        self.current_distance_from_goal = None
        self.previous_distance_from_goal = None

        self._init_robot()

        self.reset()

    def _init_robot(self):
        self.trans_field = self.supervisor.getFromDef('EPUCK').getField('translation')

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
        random.shuffle(states)
        self.initial_pos, self.final_pos = states

        self.current_distance_from_goal = self.previous_distance_from_goal = self.get_distance_from_goal()

        # set robot position
        self.trans_field.setSFVec3f([self.initial_pos[0], self.initial_pos[1], 0,])

        observation = np.array(self.lidar.getRangeImage())
        return observation, {}

    def get_current_reward(self, wd_multiplier=1, progress_multiplier=.03, time_multiplier=.01):
        wall_distance_reward = (np.abs(np.min(self.lidar.getRangeImage()) - self.WALL_DISTANCE_GOAL)) * wd_multiplier
        progress_reward = (self.previous_distance_from_goal - self.current_distance_from_goal) * progress_multiplier
        efficiency_reward = self.get_time_elapsed() * time_multiplier

        total_reward = (wall_distance_reward + progress_reward + efficiency_reward) * -1
        return total_reward

    def get_distance_from_goal(self):
        gps_readings: [float] = self.gps.getValues()
        current_pos: (float, float) = (gps_readings[0], gps_readings[1])

        return np.sqrt((current_pos[0] - self.final_pos[0]) ** 2 + (current_pos[1] - self.final_pos[1]) ** 2)

    def check_wall_collision(self, min_distance=.045):
        min_lidar_reading= np.min(self.lidar.getRangeImage())
        if min_lidar_reading < min_distance:
            return True
        return False

    def get_time_elapsed(self):
        return time.time() - self.start_time

