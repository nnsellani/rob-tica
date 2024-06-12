from webots.controllers.utils import cmd_vel

class RobotActionExecutor:
    N_ACTIONS = 5

    def __init__(self, robot, linear_vel_adjustment=.025, angular_vel_adjustment=.1):
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
        self.linear_vel = min(self.linear_vel + self.linear_vel_adjustment, 0.5)
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

        self.linear_vel = 0
        self.angular_vel = 0

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
        self.linear_vel = .4
        self.angular_vel = 0
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _rotate_right(self):
        self.linear_vel = 0
        self.angular_vel = .5
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)

    def _rotate_left(self):
        self.linear_vel = 0
        self.angular_vel = -.5
        cmd_vel(self.robot, self.linear_vel, self.angular_vel)
