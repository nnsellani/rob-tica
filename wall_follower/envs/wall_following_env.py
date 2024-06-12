import time

from gymnasium import spaces
import numpy as np

from controller import Supervisor
from wall_follower.envs import action_exectuors
from wall_follower.envs.my_env import MyEnv


class WallFollowingEnv(MyEnv):

    def __init__(self, supervisor: Supervisor, reward_multipliers=[2, .3, .05], reward_adjustment=1.0):
        self.action_executor = action_exectuors.RobotActionExecutor(supervisor)
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

    def step(self, action, end_reward=10_000):
        assert 0 <= action <= self.action_executor.N_ACTIONS - 1

        # Execute chosen action
        self.action_executor.execute(action)

        step_result = self.supervisor.step(self.STEPS_PER_ACTION)

        self.previous_distance_from_goal = self.current_distance_from_goal
        self.current_distance_from_goal = self.get_distance_from_goal()

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
            self.reward = -end_reward
            done = True
        elif self.check_max_wall_distance_crossed():
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

