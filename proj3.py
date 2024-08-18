import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Types of spaces

Discrete(3).sample() # Discrete actions

Box(0, 1, shape=(3,)).sample() # Continuous values of lower and upper

Tuple(Discrete(3), Box(0, 1, shape=(3,))).sample() #Combines spaces

Dict({'height':Discrete(2), 'speed':Box(0, 100, shape=(1,))}).sample() #

MultiBinary(4).sample() # combinations of 0 and 1

MultiDiscrete([5, 2, 2]).sample() # multidimensional discrete

# Building an environment
# Give us best shower possible between 37 and 39

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
    def step(self, action):
        self.state += action-1
        self.shower_length -= 1
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <=0 :
            done = True
        else:
            done = False
        info = {}
        return self.state, reward, done, info
    def render(self):
        None
    def reset(self):
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_length = 60
        return self.state
    
env = ShowerEnv()
env.reset()

log_path = os.path.join('training', 'logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=4000)
PPO_path = os.path.join('training', 'saved_model', 'PPO_Shower_model')
model.save(PPO_path)

evaluate_policy(model, env, n_eval_episodes=10)