import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from controller import Supervisor
from wall_following_model import WallFollowingEnv

from typing import Callable

import numpy as np


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
            self.model.save("models/ppo_wall_follower_checkpoint" + str(int(self.counter/25)))


        return True

def train_model(retrain=None):
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
        env = WallFollowingEnv(supervisor, "train")
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        n_timesteps = 614_400

        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=4,
                    n_epochs=40, learning_rate=.0005,
                    tensorboard_log='logs/PPO'
        )
        if retrain is not None:
            model = PPO.load(retrain)
            model.set_env(env)

        callback = AverageRewardCallback(check_freq=2048)

        # Train the model
        print('Training...')
        model.learn(total_timesteps=n_timesteps, callback=callback, tb_log_name='PPO')
        model.save("models/ppo_wall_follower")
        print('Training finished!')
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()

if __name__ == '__main__':
    train_model()