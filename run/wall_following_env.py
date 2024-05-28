from typing import Callable

import time

from gymnasium import spaces
import numpy as np
from gymnasium.wrappers import NormalizeObservation
from sklearn.preprocessing import normalize

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import os

#os.environ['WEBOTS_HOME'] = 'C:\Program Files\Webots'

from controller import Supervisor
from webots.controllers.utils import cmd_vel
from my_env import MyEnv


class RobotActionExecutor:

    def __init__(self, robot, vel_increase=.02, ang_increase=.1):
        self.robot = robot
        self.vel_increase = vel_increase
        self.ang_increase = ang_increase

        self.linear_vel = 0
        self.angular_vel = 0

    def action_keep_vel(self):
        #print('ACTION: KEEP VEL')
        pass

    def action_increase_linear_vel(self):
        #print('ACTION: INCREASE LINEAR VEL')
        self.linear_vel = min(self.linear_vel + self.vel_increase, 0.1)
        
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def action_decrease_linear_vel(self):
        #print('ACTION: DECREASE LINEAR VEL')
        self.linear_vel = max(self.linear_vel - self.vel_increase, 0)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def action_increase_angular_vel(self):
        #print('ACTION: INCREASE ANGULAR VEL')
        self.angular_vel = min(self.angular_vel + self.ang_increase, .5)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def action_decrease_angular_vel(self):
        #print('ACTION: DECREASE ANGULAR VEL')
        self.angular_vel = max(self.angular_vel - self.ang_increase, -.5)
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)


class WallFollowingEnv(MyEnv):

    def __init__(self, supervisor: Supervisor):
        super(WallFollowingEnv, self).__init__(supervisor)

        self.action_space = spaces.Discrete(self.N_ACTION_SPACES)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.N_OBSERVATIONS,), dtype=np.float64)

    def _init_robot(self):
        super()._init_robot()
        self.action_executor = RobotActionExecutor(self.supervisor)

    def reset(self, seed=None):
        self.action_executor.linear_vel = self.action_executor.angular_vel = 0

        observation, info = super().reset()
        observation[observation == np.inf] = 3
        observation = normalize([np.array(observation)])[0]

        return observation, info

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

        step_result = self.supervisor.step(self.STEPS_PER_ACTION)

        self.previous_distance_from_goal = self.current_distance_from_goal
        self.current_distance_from_goal = self.get_distance_from_goal()

        observation = np.array(self.lidar.getRangeImage())
        observation[observation == np.inf] = 3
        observation = normalize([observation])[0]

        reward = self.get_current_reward()
        done = (step_result == -1
                or self.current_distance_from_goal < 0.05
                or self.check_wall_collision()
                or time.time() - self.start_time >= self.TIME_LIMIT)

        if done:
            self.reset()

        # Execute action and return observation, reward, done, truncated, info
        return observation, reward, done, False, {}


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
        env = WallFollowingEnv(supervisor)
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        n_timesteps = 40_000

        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=1,
                    n_epochs=40, learning_rate=0.00005)

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
    print(supervisor)
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
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()


if __name__ == '__main__':
    run_model()