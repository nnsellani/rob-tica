import gymnasium as gym
import random
import utils

class metadata():
    supervisor=None
    robot=''
    initStates=[]

class testEnv(gym.Env):
    metadata= metadata()
    def __init__(self):
        print('test')

    def reset(self, seed=None, options=None):
        startPos=random.randint(0,7)
        utils.warp_robot(self.metadata.supervisor,self.metadata.robot,self.metadata.iniStates[startPos])

