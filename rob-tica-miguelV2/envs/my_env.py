import gymnasium
import numpy as np
import random
import time

import sys
import os

os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Supervisor, Lidar, GPS, TouchSensor


def _my_activation_func(x):
    return ((1 / (1 + np.e ** -x)) - .5) * 2


class MyEnv(gymnasium.Env):
    N_OBSERVATIONS = 180
    WALL_DISTANCE_GOAL = 0.07
    TIME_LIMIT = 60
    STEPS_PER_ACTION = 1

    INITIAL_AND_FINAL_STATES = [
        ((0.64, 0.83), (0.64, 0.11), -3.0),
        ((0.64, 0.26), (0.64, 0.68),-3.2),
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
        self.rotat_field.setSFRotation([0,0,1,angle])

        observation = self.get_my_lidar_readings()
        return observation, {}

    def get_current_reward(self, distance_from_goal_diference):
        size = self.N_OBSERVATIONS-1

        wall_distance = self.lidar.getRangeImage()[size//4]
        min_distance_index = np.argmin(self.lidar.getRangeImage())
        min_lidar_reading= np.min(self.lidar.getRangeImage())

        if min_lidar_reading <.047:
            return -1

        if min_lidar_reading >.15:
            return -1


        if min_distance_index > (3*size)//4:
            min_distance_index -=  size//4

        #print(abs((size//4) - min_distance_index))

        ang_reward = abs((size//4) - min_distance_index) / (size//2) * 0.5

        #print(abs(wall_distance - self.WALL_DISTANCE_GOAL))

        wall_distance_reward = (abs(wall_distance - self.WALL_DISTANCE_GOAL))

        total_reward = min(wall_distance_reward + ang_reward,10)
        
        return total_reward*-1 + distance_from_goal_diference*10

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

    def get_time_elapsed(self):
        return time.time() - self.start_time
