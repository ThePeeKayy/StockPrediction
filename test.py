import gym_anytrading
import gymnasium as gym
import pandas as pd
df = pd.read_csv('NETFLIX.csv')
env = gym.make('stocks-v0', df=df)
print("Max episode steps:", env.max_episode_steps)
