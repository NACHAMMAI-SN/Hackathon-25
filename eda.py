import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# Load the cleaned data
df = pd.read_csv("processed_stock_data.csv", index_col=0, parse_dates=True)
print("Data loaded successfully!")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Plot closing price
plt.figure(figsize=(12, 6))
plt.plot(df["close"], label="Close Price")
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Plot daily returns
plt.figure(figsize=(12, 6))
plt.plot(df["daily_return"], label="Daily Return", color="orange")
plt.title("Daily Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(df["close"], period=30)
decomposition.plot()
plt.show()

# Rolling statistics
df["rolling_mean"] = df["close"].rolling(window=30).mean()
df["rolling_std"] = df["close"].rolling(window=30).std()

plt.figure(figsize=(12, 6))
plt.plot(df["close"], label="Close Price")
plt.plot(df["rolling_mean"], label="Rolling Mean (30 days)", color="red")
plt.plot(df["rolling_std"], label="Rolling Std (30 days)", color="green")
plt.title("Rolling Mean and Standard Deviation")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Autocorrelation analysis
plot_acf(df["close"], lags=30)
plt.title("Autocorrelation of Closing Price")
plt.show()

# ADF test for stationarity
result = adfuller(df["close"])
print("\nADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

if result[1] <= 0.05:
    print("The time series is stationary.")
else:
    print("The time series is non-stationary.")

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot of daily returns
plt.figure(figsize=(8, 6))
sns.boxplot(df["daily_return"])
plt.title("Boxplot of Daily Returns")
plt.show()