import ta
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Polygon.io API key
POLYGON_API_KEY = 'I83HqNvUyRmSSH8sW3v0__Cox7eFXRQN'

# Step 1: Fetch stock data using Polygon.io
def fetch_stock_data(ticker):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2023-08-01/2024-08-01?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if 'results' in data:
        df = pd.DataFrame(data['results'])
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df
    else:
        raise Exception(f"Failed to fetch data for {ticker}")

# List of tickers
tickers = ['GOOGL', 'AMZN', 'MSFT', 'TSLA', 'ORCL', 'CRM', 'INTC', 'AMD', 'PYPL', 'SHOP', 'SQ']

# Fetch and store stock data for each ticker
stock_data = {}
for ticker in tickers:
    stock_data[ticker] = fetch_stock_data(ticker)

# Step 2: Function to calculate indicators for each stock
def calculate_indicators(stock_df):
    indicators = {}
    
    # Ensure the DataFrame has enough rows
    if len(stock_df) < 50:
        return None
    
    # Make an explicit copy of the DataFrame to avoid the warning
    stock_df = stock_df.copy()

    # Calculate MACD
    macd = ta.trend.MACD(stock_df['Close'])
    stock_df['MACD'] = macd.macd()
    stock_df['Signal'] = macd.macd_signal()
    stock_df['Hist'] = macd.macd_diff()
    indicators['MACD'] = stock_df['MACD'].iloc[-1]
    
    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(stock_df['Close'], window=14)
    stock_df['RSI'] = rsi.rsi()
    indicators['RSI'] = stock_df['RSI'].iloc[-1]
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(stock_df['Close'], window=20)
    stock_df['Bollinger_upper'] = bb.bollinger_hband()
    stock_df['Bollinger_lower'] = bb.bollinger_lband()
    indicators['Bollinger_position'] = (stock_df['Close'].iloc[-1] - stock_df['Bollinger_lower'].iloc[-1]) / (stock_df['Bollinger_upper'].iloc[-1] - stock_df['Bollinger_lower'].iloc[-1])
    
    return indicators



# Step 3: Prepare the dataset
X = []
y = []

for ticker, df in stock_data.items():
    if 'Close' in df.columns:
        for i in range(50, len(df)):  # We skip the first 50 days for moving averages
            indicators = calculate_indicators(df.iloc[:i])
            
            # Features
            X.append([indicators['MACD'], indicators['RSI'], indicators['Bollinger_position']])
            
            # Labels: 1 if we consider it a good buy, 0 otherwise (custom rule)
            if indicators['MACD'] > 0 and 30 < indicators['RSI'] < 70 and indicators['Bollinger_position'] < 0.2:
                y.append(1)  # Buy signal
            else:
                y.append(0)  # Do not buy

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Use the model to recommend the best stock out of 7 tickers
def recommend_best_stock(tickers, stock_data, model):
    best_stock = None
    best_score = -1
    
    for ticker in tickers:
        latest_data = calculate_indicators(stock_data[ticker].iloc[-50:])
        latest_features = np.array([[latest_data['MACD'], latest_data['RSI'], latest_data['Bollinger_position']]])
        
        # Get the predicted probability of buying the stock (confidence score)
        buy_probability = model.predict_proba(latest_features)[0][1]  # Probability for class 1 (buy)
        
        print(f"{ticker} Buy Probability: {buy_probability * 100:.2f}%")
        
        # Choose the stock with the highest probability
        if buy_probability > best_score:
            best_stock = ticker
            best_score = buy_probability
    
    return best_stock, best_score

# Step 8: Recommend the best stock
recommended_stock, score = recommend_best_stock(tickers, stock_data, model)
print(f"Recommended stock to buy: {recommended_stock} with confidence score: {score * 100:.2f}%")
