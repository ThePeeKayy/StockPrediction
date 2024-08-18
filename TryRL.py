import yfinance as yf
import pandas as pd
import gymnasium as gym
import gym_anytrading
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter

# Download stock data
df = pd.read_csv('savedGOOGL.csv')

def extract_features(df):
    df['mean_return'] = df['Close'].pct_change().mean()
    df['volatility'] = df['Close'].pct_change().std()
    df['rsi'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(window=14).mean() / 
                              df['Close'].diff().clip(upper=0).rolling(window=14).mean()))
    df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['atr'] = df['High'].subtract(df['Low']).combine(df['High'].subtract(df['Close'].shift()), max).combine(df['Low'].subtract(df['Close'].shift()), max).rolling(window=14).mean()
    df['avg_volume'] = df['Volume'].rolling(window=50).mean()
    return df[['mean_return', 'volatility', 'rsi', 'macd', 'macd_signal', 'atr', 'avg_volume', 'Close']]

# Apply the feature extraction
df = extract_features(df)
df.fillna(0, inplace=True)

# Create the environment
env = gym.make('stocks-v0', df=df)

# Check the environment
check_env(env)

# Initialize multiple models
models = {
    'DQN': DQN('MlpPolicy', env, verbose=1),
}

# Train the models
for name, model in models.items():
    model.learn(total_timesteps=4000)
    model.save(f"{name}_stock_trading")

# Load the models
models = {name: model.load(f"{name}_stock_trading") for name, model in models.items()}

# Define the neural network for DQN with LSTM and additional hidden layers
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(state_size, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # Add batch dimension if not present
        x, _ = self.lstm(state)
        x = torch.relu(self.fc1(x[:, -1, :]))  # Use the output of the last LSTM cell
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the network
state_size = env.observation_space.shape[1]  # Feature dimension size
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)

# Define optimizer and loss function
optimizer = optim.Adam(q_network.parameters(), lr=0.00042)
loss_fn = nn.MSELoss()

# Define hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.02
replay_buffer = []
batch_size = 64
max_memory_size = 5000
initial_price = df['Close'][0]

# Helper function to choose an action using the ensemble
def dynamic_weighting(models, obs, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        actions = []
        for model in models.values():
            obs = obs.reshape(-1, 2)
            action, _states = model.predict(obs, deterministic=True)
            actions.append(int(action))  # Convert action to int
        action_counts = Counter(actions)
        most_common_action = action_counts.most_common(1)[0][0]
        return most_common_action

# Function to pad states to a consistent length
def pad_state(state, target_length):
    state = np.array(state).flatten()  # Ensure the state is flattened
    if len(state) < target_length:
        return np.pad(state, (0, target_length - len(state)), 'constant')
    else:
        return state[:target_length]

# Train the model
num_episodes = 45
for episode in range(num_episodes):
    state = env.reset()
    state, info = state  # Unpacking the tuple
    state = state.reshape(1, state.shape[0], state.shape[1])  # Reshape for LSTM input
    print(f"Episode {episode + 1} - Initial state shape: {state.shape}, type: {type(state)}")
    done = False
    total_reward = 0
    step_count = 0
    while not done and step_count < 710:
        action = dynamic_weighting(models, state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        # Ensure all states have the same shape before adding to replay buffer
        next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1])  # Reshape for LSTM input
        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) > max_memory_size:
            replay_buffer.pop(0)

        state = next_state

        # Sample a batch from replay buffer
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states).squeeze(1))  # Remove extra dimension if needed
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states).squeeze(1))  # Remove extra dimension if needed
            dones = torch.FloatTensor(dones)

            # Ensure the dimensions are correct for LSTM input
            states = states.reshape(batch_size, -1, state_size)
            next_states = next_states.reshape(batch_size, -1, state_size)

            # Compute Q targets
            q_values = q_network(states).gather(1, actions)
            next_q_values = target_network(next_states).max(1)[0].detach().unsqueeze(1)
            q_targets = rewards.unsqueeze(1) + gamma * next_q_values * (1 - dones.unsqueeze(1))

            # Compute loss and optimize
            loss = loss_fn(q_values, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Test the model
test_env = gym.make('stocks-v0', df=df)
obs = test_env.reset()
obs, info = obs  # Unpacking the tuple
obs = obs.reshape(1, obs.shape[0], obs.shape[1])  # Ensure shape is (batch_size, sequence_length, input_size)

# Variables to track performance
rewards = 0
portfolio_values = []
actions = []
test_step = 0

# Run the model in the test environment
for i in range(len(test_env.df)):
    action = dynamic_weighting(models, obs, epsilon=0)
    obs, reward, done, _, info = test_env.step(action)
    obs = obs.reshape(1, obs.shape[0], obs.shape[1])  # Reshape for LSTM input
    rewards += reward
    test_step += 1
    portfolio_values.append(info['total_reward'] + initial_price)  # Assuming 'total_value' is the portfolio value in the info dictionary
    actions.append(action)
    if done or test_step > 710:
        break

# Print the total reward
print(f"Total Reward: {rewards}")

buy_and_hold_values = [initial_price]
for price in df['Close'][1:]:
    buy_and_hold_values.append(buy_and_hold_values[-1] * (price / buy_and_hold_values[-1]))

plt.figure(figsize=(14, 7))
plt.plot(buy_and_hold_values, color='blue', label='Buy and Hold Strategy', linestyle='--')

plt.plot(portfolio_values, color='red', label='Ensemble Portfolio Value')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()

# Optional: Plot the actions taken over time
plt.figure(figsize=(14, 7))
plt.plot(actions, label='Actions Taken')
plt.xlabel('Time Steps')
plt.ylabel('Action')
plt.title('Actions Over Time')
plt.legend()
plt.show()
