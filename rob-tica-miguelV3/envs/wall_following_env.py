from typing import Callable

import time

from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

import os

#os.environ['WEBOTS_HOME'] = '/usr/local/webots'

from controller import Supervisor, Motor
from controllers.utils import cmd_vel
from my_env import MyEnv


class RobotActionExecutor:
    N_ACTIONS = 3

    def __init__(self, robot, linear_vel_adjustment=.025, angular_vel_adjustment=.25):
        
        self.robot = robot
        self.linear_vel_adjustment = linear_vel_adjustment
        self.angular_vel_adjustment = angular_vel_adjustment

        self.linear_vel = 0
        self.angular_vel = 0



    def execute(self, action):
        if self.linear_vel != .05:
            self.linear_vel = 0.05
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)
        if action == 0:
            self._keep_vel()
        elif action == 1:
            self._increase_angular_vel()
        elif action == 2:
            self._decrease_angular_vel()

    def _keep_vel(self):
        #print('ACTION: KEEP VEL')
        pass

    def _move(self):
        #print('ACTION: INCREASE LINEAR VEL')
        if self.linear_vel != 0.1:
            self.linear_vel = 0.1
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _stop(self):
        if self.linear_vel != 0:
            self.linear_vel = 0
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _increase_angular_vel(self):
        #print('ACTION: INCREASE ANGULAR VEL')
        if self.angular_vel != 1:
            self.angular_vel = min(self.angular_vel + self.angular_vel_adjustment, 1)
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _decrease_angular_vel(self):
        #print('ACTION: DECREASE ANGULAR VEL')
        if self.angular_vel != -1:
            self.angular_vel = max(self.angular_vel - self.angular_vel_adjustment, -1)
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)


class RobotActionExecutor2:
    N_ACTIONS = 4

    def __init__(self, robot):
        self.robot = robot
        self.linear = 0
        self.current_angular_velocity = 0

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
        self.linear_vel = .05
        self.angular_vel = 0
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _rotate_right(self):
        if self.angular_vel != 1:
            self.linear_vel = .05
            self.angular_vel = 1
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _rotate_left(self):
        if self.angular_vel != -1:
            self.linear_vel = .05
            self.angular_vel = -1
            cmd_vel(self.robot, self.linear_vel, self.angular_vel)

class WallFollowingEnv(MyEnv):

    def __init__(self, supervisor: Supervisor, reward_multipliers=[2, .3, .05], reward_adjustment=1.0):
        self.action_executor = RobotActionExecutor2(supervisor)
        self.action_space = spaces.Discrete(self.action_executor.N_ACTIONS)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.N_OBSERVATIONS + 2,), dtype=np.float64)

        super(WallFollowingEnv, self).__init__(supervisor)

        self.reward_multipliers = [x * reward_adjustment for x in reward_multipliers]
        self.reward = None

    def _init_robot(self):
        super()._init_robot()

    def _normalize_observation(self, observation):
        return np.divide(observation, 3)

    def get_observation(self):
        return np.concatenate((self._normalize_observation(self.get_my_lidar_readings()),
                               [self.action_executor.linear_vel, self.action_executor.angular_vel],))

    def reset(self, seed=None):
        self.action_executor.linear_vel = self.action_executor.angular_vel = 0
        self.reward = 0
        self.current_location = self.gps.getValues()[0:2]

        super().reset()
        observation = self.get_observation()

        return observation, {}

    def step(self, action, end_reward=50):
        assert 0 <= action <= self.action_executor.N_ACTIONS - 1

        # Execute chosen action
        self.action_executor.execute(action)

        step_result = self.supervisor.step(self.STEPS_PER_ACTION)

        self.previous_distance_from_goal = self.current_distance_from_goal
        self.current_distance_from_goal = self.get_distance_from_goal()

        distance_from_goal_diference = self.previous_distance_from_goal - self.current_distance_from_goal

        self.previous_location = self.current_location 
        self.current_location = self.gps.getValues()[0:2]

        observation = self.get_observation()

        if step_result == -1:
            self.reward = -end_reward
            done = True
        elif self.current_distance_from_goal < 0.08:
            self.reward = end_reward
            done = True
        elif self.check_wall_collision():
            self.reward = -end_reward*10
            done = True
        elif time.time() - self.start_time > self.TIME_LIMIT:
            self.reward = -end_reward
            done = True
        else:
            self.reward = self.get_current_reward(distance_from_goal_diference)
            done = False

        # Execute action and return observation, reward, done, truncated, info
        return observation, self.reward, done, False, {}

class AverageRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=2):
        super(AverageRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.counter = 0

    def _on_step(self) -> bool:
        # Collect rewards
        if len(self.locals['infos']) > 0:
            reward = self.locals['rewards']
            if isinstance(reward, np.ndarray):
                reward = reward.item()  # Convert NumPy array to scalar
            self.episode_rewards.append(reward)

        # Print average reward at check frequency
        if self.n_calls % self.check_freq == 0:
            self.counter +=1
            if len(self.episode_rewards) > 0:
                average_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                print(f"Step: {self.n_calls}, Average Reward: {average_reward:.2f}")
                self.episode_rewards = []  # Reset for next interval

        if self.counter % 25 == 0:
            os.makedirs("checkpoint", exist_ok=True)
            self.model.save("ppo_wall_follower_checkpoint" + str(int(self.counter/25)))


        return True

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
        env = WallFollowingEnv(supervisor, reward_multipliers=(2, .3, .1), reward_adjustment=1)
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        n_timesteps = 50_000

        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=4,
                    n_epochs=40, learning_rate=.0005,
                    tensorboard_log='logs/PPO'
                    #policy_kwargs={
                    #    'net_arch': [512, 512, 256, 128, 32]
                    #}
        )

        model = PPO.load('ppo_wall_follower_checkpoint_9_retrain.zip')
        model.set_env(env)

        callback = AverageRewardCallback(check_freq=2048)

        # Train the model
        print('Training...')
        model.learn(total_timesteps=n_timesteps, callback=callback, tb_log_name='PPO')
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

        model = PPO.load('ppo_wall_follower_checkpoint2.zip')

        print('Testing...')
        # Evaluate the model
        obs = env.reset()
        every_n = 100
        count = every_n

        while True:
            action, _states = model.predict(obs)

            print(action)

            count -= 1
            if count == 0:
                count = every_n

                print(f'linear vel: {env.envs[0].action_executor.linear_vel:.3f}\t' +
                      f'angular vel: {env.envs[0].action_executor.angular_vel:.3f}')

                print(np.min(env.envs[0].get_my_lidar_readings()[20:41]))

            obs, rewards, dones, info = env.step(action)
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()


if __name__ == '__main__':
    train_model()
