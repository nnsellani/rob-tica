from typing import Callable

import time

from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Supervisor
from webots.controllers.utils import cmd_vel
from envs.my_env import MyEnv


class RobotActionExecutor:

    N_ACTIONS = 5

    def __init__(self, robot, linear_vel_adjustment=.03, angular_vel_adjustment=.15):
        self.robot = robot
        self.linear_vel_adjustment = linear_vel_adjustment
        self.angular_vel_adjustment = angular_vel_adjustment

        self.linear_vel = 0
        self.angular_vel = 0

    def execute(self, action):
        if action == 0:
            self._keep_vel()
        elif action == 1:
            self._increase_linear_vel()
        elif action == 2:
            self._decrease_linear_vel()
        elif action == 3:
            self._increase_angular_vel()
        elif action == 4:
            self._decrease_angular_vel()

    def _keep_vel(self):
        #print('ACTION: KEEP VEL')
        pass

    def _increase_linear_vel(self):
        #print('ACTION: INCREASE LINEAR VEL')
        self.linear_vel = min(self.linear_vel + self.linear_vel_adjustment, 0.1)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _decrease_linear_vel(self):
        #print('ACTION: DECREASE LINEAR VEL')
        self.linear_vel = max(self.linear_vel - self.linear_vel_adjustment, 0)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _increase_angular_vel(self):
        #print('ACTION: INCREASE ANGULAR VEL')
        self.angular_vel = min(self.angular_vel + self.angular_vel_adjustment, .6)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _decrease_angular_vel(self):
        #print('ACTION: DECREASE ANGULAR VEL')
        self.angular_vel = max(self.angular_vel - self.angular_vel_adjustment, -0.6)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)


class RobotActionExecutor2:

    N_ACTIONS = 4

    def __init__(self, robot):
        self.robot = robot

    def execute(self, action):
        if action == 0:
            self._keep_action()
        elif action == 1:
            self._move_forward()
        elif action == 2:
            self._rotate_right()
        elif action == 3:
            self._rotate_left()

    def _keep_action(self):
        pass

    def _move_forward(self):
        cmd_vel(self.robot, .4, 0)

    def _rotate_right(self):
        cmd_vel(self.robot, 0, .5)

    def _rotate_left(self):
        cmd_vel(self.robot, 0, .5)


class WallFollowingEnv(MyEnv):

    def __init__(self, supervisor: Supervisor, reward_multipliers=[2, .3, .05], reward_adjustment=1.0):
        self.action_executor = RobotActionExecutor(supervisor)
        self.action_space = spaces.Discrete(self.action_executor.N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N_OBSERVATIONS,), dtype=np.float64)

        super(WallFollowingEnv, self).__init__(supervisor)

        self.reward_multipliers = [x * reward_adjustment for x in reward_multipliers]
        self.reward = None

    def _init_robot(self):
        super()._init_robot()

    def _normalize_observation(self, observation):
        return np.divide(observation, 3)

    def reset(self, seed=None):
        self.action_executor.linear_vel = self.action_executor.angular_vel = 0
        self.reward = 0

        observation, info = super().reset()
        observation = self._normalize_observation(observation)

        return observation, info

    def step(self, action, end_reward=1):
        assert 0 <= action <= self.action_executor.N_ACTIONS - 1

        # Execute chosen action
        self.action_executor.execute(action)

        step_result = self.supervisor.step(self.STEPS_PER_ACTION)

        self.previous_distance_from_goal = self.current_distance_from_goal
        self.current_distance_from_goal = self.get_distance_from_goal()

        observation = self._normalize_observation(self.get_my_lidar_readings())

        if step_result == -1:
            self.reward = -end_reward
            done = True
        elif self.current_distance_from_goal < 0.1:
            self.reward = end_reward
            done = True
        elif self.check_wall_collision():
            self.reward = -end_reward
            done = True
        elif time.time() - self.start_time > self.TIME_LIMIT:
            self.reward = -end_reward
            done = True
        else:
            self.reward = self.get_current_reward(*self.reward_multipliers)
            done = False

        # Execute action and return observation, reward, done, truncated, info
        return observation, self.reward, done, False, {}


def train_model():
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
          current learning rate depending on remaining progress
        """

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func


    supervisor = Supervisor()
    try:
        env = WallFollowingEnv(supervisor, reward_multipliers=(10, .3, .05), reward_adjustment=.01)
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        n_timesteps = 75_000

        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=1,
                    n_epochs=5, learning_rate=.001,
                    #policy_kwargs={
                    #    'net_arch': [512, 512, 128, 32]
                    #}
        )

        # Train the model
        print('Training...')
        model.learn(total_timesteps=n_timesteps)
        model.save("ppo_wall_follower")
        print('Training finished!')
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()


def run_model():
    supervisor = Supervisor()
    try:
        env = WallFollowingEnv(supervisor)
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        model = PPO.load('ppo_wall_follower.zip')

        print('Testing...')
        # Evaluate the model
        obs = env.reset()
        every_n = 100
        count = every_n
        while True:
            action, _states = model.predict(obs)

            count -= 1
            if count == 0:
                count = every_n
                print(f'linear vel: {env.envs[0].action_executor.linear_vel:.3f}\t' +
                      f'angular vel: {env.envs[0].action_executor.angular_vel:.3f}')

            obs, rewards, dones, info = env.step(action)
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()


if __name__ == '__main__':
    train_model()
