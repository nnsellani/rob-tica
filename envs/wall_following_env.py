import gymnasium
from gymnasium import spaces
import numpy as np
import random
import time

from gymnasium.wrappers import NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

from reward import calculate_reward

import sys
import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Lidar, GPS, Supervisor, TouchSensor
from webots.controllers.utils import cmd_vel
from env.my_env import MyEnv


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


class WallFollowingEnv(MyEnv):

    def __init__(self, supervisor: Supervisor):
        super(WallFollowingEnv, self).__init__(supervisor)

        self.action_space = spaces.Discrete(self.N_ACTION_SPACES)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.N_OBSERVATIONS,), dtype=np.float64)

    def _init_robot(self):
        super()._init_robot()
        self.action_executor = RobotActionExecutor(self.supervisor)

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
        reward = self.get_current_reward()
        done = (step_result == -1
                or self.current_distance_from_goal < 0.05
                or self.touch_sensor.getValue() == 1)

        if done:
            self.reset()

        # Execute action and return observation, reward, done, truncated, info
        return observation, reward, done, False, {}


if __name__ == '__main__':
    supervisor = Supervisor()
    try:
        env = WallFollowingEnv(supervisor)
        #env = NormalizeObservation(env)
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