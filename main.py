import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# load env

# env = gym.make('CartPole-v0')
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print(f'Episode: {episode} | Score: {score}')

#     env.close()

log_path = os.path.join('Training', 'Logs')
env = gym.make('CartPole-v0', render_mode='human')
env = DummyVecEnv([lambda : env]) #wrapper for non vectorized algo
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000) #lower time steps for easy tasks

# save and reload model

PPO_Path = os.path.join('Training', 'saved_models', 'PPO_Model_Cartpole')
model.save(PPO_Path)

# Reload model

model = PPO.load(PPO_Path, env=env)

# Evaluation matrix

evaluate_policy(model, env, n_eval_episodes=10, render=True)

# Test model

for episode in range(1, 6):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action, _ = model.predict(obs) #Now using Model
        obs, reward, done, info = env.step(action)
        score += reward
    print(f'Episode:{episode} Score:{score}')

# View logs in TensorBoard

training_log_path = os.path.join('training', 'PPO_2')

# adding callback to training stage

save_path = os.path.join('training', 'saved_models')
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
# create function that stops training once reward threshold hit
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model=save_path, verbose=1)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(time_steps = 20000, callback=eval_callback)

# change policy

net_arch = [dict(pi=[128, 128, 128, 128], vf = [128, 128, 128, 128])]
model = PPO('MlPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch})
model.learn(time_steps = 20000, callback=eval_callback)

# change algorithm

from stable_baselines3 import DQN
model = DQN('MlPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(time_steps = 20000)
