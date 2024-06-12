from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from controller import Supervisor
from wall_following_model import WallFollowingEnv

import numpy as np

def run_model(model,map,testing):
    supervisor = Supervisor()
    try:
        env = WallFollowingEnv(supervisor,map,testing)
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        model = PPO.load(model)

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

                print(np.min(env.envs[0].get_my_lidar_readings()[20:41]))

            obs, rewards, dones, info = env.step(action)
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()


if __name__ == '__main__':
    run_model("models/ppo_wall_follower_model_D.zip", "maze2", True)
