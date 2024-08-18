import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')

for episode in range(1):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
    print(f'Episode:{episode} Score:{score}')

env.close()

# Vectorise environment and train model, use 4 environments together

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
log_path = os.path.join('training', 'logs')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path, device='cuda')
model.learn(total_timesteps=100000)

# Save path

a2c_path = os.path.join('training', 'saved_model', 'a2c_breakout')

# Evaluate ( can only eval single model)

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
