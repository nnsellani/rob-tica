import gymnasium
from gymnasium import spaces
import numpy as np
import random
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from sklearn import preprocessing as pre

import sys
import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Robot, Lidar, GPS, Supervisor, TouchSensor
from webots.controllers.utils import cmd_vel



class RobotActionExecutor:

    def __init__(self, robot, vel_increase=.05):
        self.robot = robot
        self.vel_increase = vel_increase

        self.linear_vel = 0
        self.angular_vel = 0

    def action_keep_vel(self):
        print('ACTION: KEEP VEL')

    def action_increase_linear_vel(self):
        print('ACTION: INCREASE LINEAR VEL')
        self.linear_vel = min(self.linear_vel + self.vel_increase, .3)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def action_decrease_linear_vel(self):
        print('ACTION: DECREASE LINEAR VEL')
        self.linear_vel = max(self.linear_vel - self.vel_increase, 0)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def action_increase_angular_vel(self):
        print('ACTION: INCREASE ANGULAR VEL')
        self.angular_vel = min(self.angular_vel + self.vel_increase, .5)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def action_decrease_angular_vel(self):
        print('ACTION: DECREASE ANGULAR VEL')
        self.angular_vel = max(self.angular_vel - self.vel_increase, 0)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)


class WallFollowingEnv(gymnasium.Env):

    N_ACTION_SPACES = 5
    N_OBSERVATIONS = 180
    TIME_PENALTY_FACTOR = 1

    # (x, y)
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

    def __init__(self, supervisor, wall_distance_goal=0.05):
        super(WallFollowingEnv, self).__init__()

        self.supervisor = supervisor

        self.start_time = None
        self.initial_pos = None
        self.final_pos = None
        self.current_distance_from_goal = None
        self.previous_distance_to_final = None

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.N_ACTION_SPACES)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.N_OBSERVATIONS,), dtype=np.float64)

        self.wall_distance_goal = wall_distance_goal

        self._init_robot()

        self.reset()

    def _init_robot(self):
        self.trans_field = self.supervisor.getFromDef('EPUCK').getField('translation')

        self.lidar: Lidar = self.supervisor.getDevice('lidar')
        self.gps: GPS = self.supervisor.getDevice('gps')

        timestep: int = int(self.supervisor.getBasicTimeStep())  # in ms

        self.lidar.enable(timestep)
        self.lidar.enablePointCloud()
        print(self.lidar.getRangeImage())

        self.gps.enable(timestep)
        print(self.gps.getValues())

        self.touch_sensor: TouchSensor = self.supervisor.getDevice('touch sensor')
        self.touch_sensor.enable(timestep)

        self.action_executor = RobotActionExecutor(self.supervisor)

    def reset(self, seed=None):
        self.start_time = time.time()

        states = list(random.choice(self.INITIAL_AND_FINAL_STATES))
        random.shuffle(states)
        self.initial_pos, self.final_pos = states

        self.current_distance_from_goal = self.previous_distance_to_final = self.get_distance_from_goal()

        # set robot position
        self.trans_field.setSFVec3f([self.initial_pos[0], self.initial_pos[1], 0,])

        observation = np.array(self.lidar.getRangeImage())
        return observation, {}

    def step(self, action):
        assert 0 <= action <= self.N_ACTION_SPACES - 1

        # Execute chosen action
        if action == 0:
            self.action_executor.action_keep_vel()
        elif action == 1:
            self.action_executor.action_increase_linear_vel()
        elif action == 2:
            self.action_executor.action_decrease_linear_vel()
        elif action == 3:
            self.action_executor.action_increase_angular_vel()
        elif action == 4:
            self.action_executor.action_decrease_angular_vel()

        step_result = self.supervisor.step()

        self.previous_distance_to_final = self.current_distance_from_goal
        self.current_distance_from_goal = self.get_distance_from_goal()

        observation = np.array(self.lidar.getRangeImage())
        reward = self.get_reward()
        done = (step_result == -1
                or self.current_distance_from_goal < 0.05
                or self.touch_sensor.getValue() == 1)

        #if done:
            #self.reset()

        # Execute action and return observation, reward, done, truncated, info
        return observation, reward, done, False, {}

    def get_distance_from_goal(self):
        gps_readings: [float] = self.gps.getValues()
        current_pos: (float, float) = (gps_readings[0], gps_readings[1])

        return np.sqrt((current_pos[0] - self.final_pos[0]) ** 2 + (current_pos[1] - self.final_pos[1]) ** 2)

    def get_time_elapsed(self):
        return time.time() - self.start_time

    def get_reward(self):
        wall_distance_reward = 1 - (min(self.lidar.getRangeImage()) - self.wall_distance_goal)
        progress_reward = self.previous_distance_to_final - self.current_distance_from_goal
        #efficiency_reward = -self.get_time_elapsed() * self.TIME_PENALTY_FACTOR

        total_reward = wall_distance_reward + progress_reward #+ efficiency_reward
        return total_reward


if __name__ == '__main__':
    supervisor = Supervisor()
    try:
        env = WallFollowingEnv(supervisor)
        check_env(env)

        '''
        env.reset()
        while True:
            env.step(1)
        '''

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=1)

        # Train the model
        model.learn(total_timesteps=10000)

        # Evaluate the model
        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()