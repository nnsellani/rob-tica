import time

from gymnasium import spaces
import numpy as np

from controller import Supervisor


from utils import cmd_vel
from env import MyEnv


class RobotActionExecutor_incremental:
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


class RobotActionExecutor_total:
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

    wall_distance_list=[]
    wall_distance_average_list=[]
    time_list=[]
    success_list=[]

    def __init__(self, supervisor: Supervisor, map, testing=False):
        self.testing = testing
        self.action_executor = RobotActionExecutor_total(supervisor)
        self.action_space = spaces.Discrete(self.action_executor.N_ACTIONS)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.N_OBSERVATIONS + 2,), dtype=np.float64)

        super(WallFollowingEnv, self).__init__(supervisor,map)

        self.reward = None

    def _init_robot(self):
        super()._init_robot()

    def _normalize_observation(self, observation):
        return np.divide(observation, 3)

    def get_observation(self):
        return np.concatenate((self._normalize_observation(self.get_my_lidar_readings()),
                               [self.action_executor.linear_vel, self.action_executor.angular_vel],))

    def reset(self, seed=None):

        if self.start_time and self.testing:
            if (time.time() - self.start_time) > 1:
                self.time_list.append(time.time() - self.start_time)
                self.wall_distance_average_list.append(np.average(self.wall_distance_list))
                self.wall_distance_list = []

        if len(self.success_list) == 10:
            print(self.time_list)
            print(self.wall_distance_average_list)
            print(self.success_list)
            exit()

        self.action_executor.linear_vel = self.action_executor.angular_vel = 0
        self.reward = 0
        self.current_location = self.gps.getValues()[0:2]

        super().reset()
        observation = self.get_observation()

        return observation, {}

    def step(self, action, end_reward=50):
        assert 0 <= action <= self.action_executor.N_ACTIONS - 1

        if self.testing:
            self.wall_distance_list.append(self.lidar.getRangeImage()[self.N_OBSERVATIONS//4])

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
            if self.testing:
                self.success_list.append(-1)
            self.reward = -end_reward
            done = True
        elif self.current_distance_from_goal < 0.08:
            self.reward = end_reward
            done = True
        elif self.current_distance_from_goal < 0.2 and self.testing:
            self.success_list.append(1)
            self.reward = end_reward
            done = True
        elif self.check_wall_collision():
            if self.testing:
                self.success_list.append(-1)
            self.reward = -end_reward*10
            done = True
        elif (time.time() - self.start_time > self.TIME_LIMIT) and not self.testing:
            self.reward = -end_reward
            done = True
        else:
            self.reward = self.get_current_reward(distance_from_goal_diference)
            done = False

        # Execute action and return observation, reward, done, truncated, info
        return observation, self.reward, done, False, {}






