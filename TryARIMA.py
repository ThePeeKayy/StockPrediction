import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('bigNVDA.csv', parse_dates=['Date'], index_col='Date')

# Ensure the data is sorted by date
data = data.sort_index()

# Split the data into training and testing sets
train_data = data[:-360]
test_data = data[-360:-240]

# Define the ARIMA model
p, d, q = 34, 4, 18 #34, 4, 18
model = SARIMAX(train_data['Close'], order=(p, d, q), seasonal_order=(0, 0, 0, 0))

# Fit the model
model_fit = model.fit(disp=False)

# Make predictions for the test set
forecast_steps = 120
forecast = model_fit.get_forecast(steps=forecast_steps)

# Extract the predicted mean and confidence intervals
forecast_df = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Align the forecast index with the test data
forecast_df.index = test_data.index
forecast_ci.index = test_data.index

# Plot the forecast vs actual values
plt.figure(figsize=(10, 6))
plt.plot(test_data['Close'], label='Actual Test Data', color='blue')
plt.plot(forecast_df, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.4)
plt.title('Nvidia Stock Price Forecast vs Actual')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.show()

# Calculate and print the Mean Squared Error
mse = mean_squared_error(test_data['Close'], forecast_df)
print(f'Mean Squared Error: {mse}')
