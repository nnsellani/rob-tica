from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import numpy as np

import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'
from controller import Supervisor
from wall_follower.envs.wall_following_env import WallFollowingEnv


if __name__ == '__main__':

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

            #print(action)

            count -= 1
            if count == 0:
                count = every_n

                #print(f'linear vel: {env.envs[0].action_executor.linear_vel:.3f}\t' +
                #      f'angular vel: {env.envs[0].action_executor.angular_vel:.3f}')

                print(env.envs[0].get_current_reward(3, 1.5, 1.2))
                print(np.min(env.envs[0].get_my_lidar_readings()[28:33]), end='\n\n')

            obs, rewards, dones, info = env.step(action)
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()