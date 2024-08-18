import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

env = gym.make('CarRacing-v2', render_mode="human")
env.reset()
# for episode in range(1):
#     obs = env.reset()
#     done = False
#     score = 0
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         score += reward
#     print(f'Episode:{episode} Score:{score}')

# env.close()

env = DummyVecEnv([lambda: env])
log_path = os.path.join('training', 'logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path, device='cuda')
model.learn(total_timesteps=150000)
ppo_path = os.path.join('training', 'saved_models', 'PPO_Driving_model')
model.save(ppo_path)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
