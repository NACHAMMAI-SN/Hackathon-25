import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the cleaned data (non-stationary)
df = pd.read_csv("processed_stock_data.csv", index_col=0, parse_dates=True)
print("Data loaded successfully!")
print(df.head())

# Ensure the index is monotonic and has a frequency
df = df.sort_index()  # Sort the index to make it monotonic
df = df.asfreq('D')   # Set the frequency to daily ('D')

# Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Fill missing values using interpolation
df = df.interpolate()

print("\nMissing values after handling:")
print(df.isnull().sum())

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Debugging: Check train and test sets
print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(train.head())
print(test.head())

# Check stationarity
def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("Series is not stationary. Apply differencing or transformations.")
    else:
        print("Series is stationary.")

print("\nChecking stationarity of training data:")
check_stationarity(train["close"])

# Apply differencing if necessary
train_diff = train["close"].diff().dropna()
print("\nChecking stationarity of differenced training data:")
check_stationarity(train_diff)

# Plot differenced data
plt.figure(figsize=(12, 6))
plt.plot(train_diff, label="Differenced Close Price")
plt.title("Differenced Close Price")
plt.xlabel("Date")
plt.ylabel("Differenced Close Price")
plt.legend()
plt.show()

# Fit ARIMA model with exogenous variables
try:
    model = ARIMA(train["close"], order=(2, 1, 2), exog=train[["open", "high", "low", "volume"]])
    model_fit = model.fit()
    print(model_fit.summary())
except Exception as e:
    print(f"Error fitting ARIMA model: {e}")
    raise

# Extend exogenous variables for the next 10 days using a rolling average
exog_test = test[["open", "high", "low", "volume"]]
exog_future = exog_test.rolling(window=10).mean().iloc[-10:]
exog_forecast = pd.concat([exog_test, exog_future])

# Forecast on the test set and the next 10 days
try:
    forecast_result = model_fit.get_forecast(steps=len(test) + 10, exog=exog_forecast)
    forecast = forecast_result.predicted_mean
    forecast_index = pd.date_range(start=test.index[0], periods=len(test) + 10, freq='D')
    forecast = pd.Series(forecast, index=forecast_index)
except Exception as e:
    print(f"Error generating forecast: {e}")
    raise

# Check the forecast object
print("\nForecast object:")
print(forecast)

# Handle NaN values in test and forecast
test = test.dropna()
forecast = forecast.dropna()

# Debugging: Check lengths of test and forecast
print(f"\nLength of test set: {len(test)}")
print(f"Length of forecast: {len(forecast)}")

if len(test) == 0 or len(forecast) == 0:
    raise ValueError("Test or forecast series is empty after dropping NaN values.")

# Calculate evaluation metrics for the test set
test_forecast = forecast[:len(test)]  # Forecast for the test set
mae = mean_absolute_error(test["close"], test_forecast)
rmse = np.sqrt(mean_squared_error(test["close"], test_forecast))  # RMSE
mape = np.mean(np.abs((test["close"] - test_forecast) / test["close"])) * 100  # MAPE

print(f"ARIMA Model Metrics:")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")

# Plot actual vs. forecasted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["close"], label="Actual", color="blue")
plt.plot(test.index, test_forecast, label="Forecast (Test Set)", color="red")
plt.plot(forecast.index[len(test):], forecast[len(test):], label="Forecast (Next 10 Days)", color="green", linestyle="--")
plt.title("Actual vs. Forecasted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()