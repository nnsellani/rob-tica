from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import os
os.environ['WEBOTS_HOME'] = '/usr/local/webots'
from controller import Supervisor
from wall_follower.envs.wall_following_env import WallFollowingEnv


if __name__ == '__main__':
    supervisor = Supervisor()
    try:
        env = WallFollowingEnv(supervisor, reward_multipliers=(2, .5, .4), reward_adjustment=3)
        #env = NormalizeObservation(env)
        check_env(env)

        # Wrap the environment
        env = DummyVecEnv([lambda: env])

        n_timesteps = 100_000

        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=1,
                    n_epochs=10, learning_rate=.001,
                    tensorboard_log='logs/PPO'
                    #policy_kwargs={
                    #    'net_arch': [512, 512, 256, 128, 32]
                    #}
        )

        #eval_callback = EvalCallback(env, best_model_save_path='logs/best_model/',
        #                             log_path='logs/results/', eval_freq=500,
        #                             deterministic=True, render=False)

        # Train the model
        print('Training...')
        model.learn(total_timesteps=n_timesteps, tb_log_name='PPO')
        model.save("ppo_wall_follower")
        print('Training finished!')
    except Exception as e:
        raise e
    finally:
        supervisor.__del__()