import yfinance as yf
import pandas as pd
import gymnasium as gym
import gym_anytrading
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from stable_baselines3.common.env_checker import check_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter
import torch.nn.functional as F
import ta
from ta.trend import IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import ROCIndicator
from ta.trend import CCIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import PSARIndicator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 3 # 8 ok
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('bigWMT.csv')
test_df = df[2200:2400]
test_df = pd.concat([test_df] * 12, ignore_index=True)
df = df[-2400:].reset_index(drop=True)

print(len(df))

def extract_features(df):
    # Mean Return and Volatility
    df['mean_return'] = df['Close'].pct_change().mean()
    df['volatility'] = df['Close'].pct_change().std()

    # Relative Strength Index (RSI)
    df['rsi'] = RSIIndicator(df['Close']).rsi()

    # MACD and MACD Signal
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()

    # Average Volume
    df['avg_volume'] = df['Volume'].rolling(window=50).mean()

    # Bollinger Bands
    bollinger = BollingerBands(df['Close'])
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()

    # Stochastic Oscillator
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['stochastic_k'] = stoch.stoch()
    df['stochastic_d'] = stoch.stoch_signal()

    # Exponential Moving Average (EMA)
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    # Momentum
    df['momentum'] = ROCIndicator(df['Close'], window=10).roc()

    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()

    # Parabolic SAR
    psar = PSARIndicator(df['High'], df['Low'], df['Close'])
    df['psar'] = psar.psar()

    # Average Directional Index (ADX)
    adx = ADXIndicator(df['High'], df['Low'], df['Close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()

    # Commodity Channel Index (CCI)
    df['cci'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci()

    # Average True Range (ATR)
    df['atr'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    return df[['mean_return', 'volatility', 'rsi', 'macd', 'macd_signal', 'avg_volume',
               'bb_middle', 'bb_high', 'bb_low', 'stochastic_k', 'stochastic_d', 'ema_12',
               'momentum', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base_line', 'ichimoku_conversion_line',
               'psar', 'adx', 'adx_pos', 'adx_neg', 'cci', 'atr', 'Close']]

df = extract_features(df)
df.dropna(inplace=True)
df = df.reset_index(drop=True)

test_df = extract_features(test_df)
test_df.dropna(inplace=True)
test_df = test_df.reset_index(drop=True)

env = gym.make('stocks-v0', df=df)

check_env(env)

# model = QRDQN('MlpPolicy', env, buffer_size=35000, learning_rate=0.0003, verbose=1, device=device)
# model.learn(total_timesteps=600000)  # Adjusted timesteps
# model.save("Walmart_QRDQN_stock_trading")

# Load the model
model = QRDQN.load("Walmart_QRDQN_stock_trading", env=env, device=device)

# Define the neural network for DQN with LSTM and additional hidden layers
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, lstm_hidden_size=128, fc_units=256, dropout_prob=0.4):
        super(QNetwork, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(state_size, lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # Add batch dimension if not present
        x, _ = self.lstm(state)
        x = x[:, -1, :]  # Use the output of the last LSTM cell
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

# Initialize the network
state_size = env.observation_space.shape[1]  # Feature dimension size
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size).to(device)
target_network = QNetwork(state_size, action_size).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(q_network.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

# Define hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9  # 0.997
min_epsilon = 0.015 
batch_size = 128
initial_price = df['Close'][0]

# Train the model
num_episodes = 10
for episode in range(num_episodes):
    state, info = env.reset()
    random_steps = random.randint(0, len(env.df) - 650)
    for _ in range(random_steps):
        state, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            state, info = env.reset()
            
    state = state.reshape(1, state.shape[0], state.shape[1])  # Reshape for LSTM input
    state = torch.FloatTensor(state).to(device)  # Move state to device
    print(f"Episode {episode + 1} - Initial state shape: {state.shape}, type: {type(state)}")
    done = False
    total_reward = 0
    step_count = 0
    while not done and step_count < 600:
        if random.random() < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            obs_cpu = state.cpu().numpy()  # Move to CPU and convert to numpy
            action, _states = model.predict(obs_cpu, deterministic=True)
            action = int(action)  # Convert action to int

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1])  # Reshape for LSTM input
        next_state = torch.FloatTensor(next_state).to(device)  # Move next_state to device

        state = next_state

        if model.replay_buffer.size() >= batch_size:
            replay_data = model.replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = replay_data.observations, replay_data.actions, replay_data.rewards, replay_data.next_observations, replay_data.dones

            states = torch.FloatTensor(np.array(states).squeeze(1)).to(device)  # Remove extra dimension if needed
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states).squeeze(1)).to(device)  # Remove extra dimension if needed
            dones = torch.FloatTensor(dones).to(device)

            states = states.reshape(batch_size, -1, state_size)
            next_states = next_states.reshape(batch_size, -1, state_size)

            q_values = q_network(states).gather(1, actions)
            next_q_values = target_network(next_states).max(1)[0].detach().unsqueeze(1)
            q_targets = rewards.unsqueeze(1) + gamma * next_q_values * (1 - dones.unsqueeze(1))

            loss = loss_fn(q_values, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Test the model
test_env = gym.make('stocks-v0', df=test_df)
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
    if not isinstance(obs, torch.Tensor):
        obs = torch.FloatTensor(obs).to(device)
    action, _states = model.predict(obs.cpu().numpy(), deterministic=True)
    action = int(action)  # Convert action to int
    obs, reward, done, _, info = test_env.step(action)
    obs = obs.reshape(1, obs.shape[0], obs.shape[1])  # Reshape for LSTM input
    obs = torch.FloatTensor(obs).to(device)  # Move obs to device
    rewards += reward
    test_step += 1
    portfolio_values.append(info['total_reward'] + initial_price)  # Assuming 'total_value' is the portfolio value in the info dictionary
    actions.append(action)
    if test_step > 2300:
        break

# Print the total reward
print(f"Walmart Total Reward: {rewards}")
test_df = test_df[:2301]
buy_and_hold_values = [initial_price]
cumulative_returns = (1 + test_df['Close'].pct_change().fillna(0)).cumprod()

for price in cumulative_returns[1:]:
    buy_and_hold_values.append(initial_price * price)

plt.figure(figsize=(14, 7))
print(f'index: {test_df.index}')
print(len(buy_and_hold_values))
print(len(portfolio_values[:len(test_df)]))
plt.plot(test_df.index, test_df['Close'], label='Actual Close Prices')
plt.plot(test_df.index, buy_and_hold_values[:2301], color='blue', label='Buy and Hold Strategy', linestyle='--')
plt.plot(test_df.index, portfolio_values[:len(test_df)], color='red', label='Ensemble Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()
