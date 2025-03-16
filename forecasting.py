import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Simulate ARIMA training and evaluation functions
def train_arima_model(df):
    # Fit an ARIMA model (simple example)
    model = ARIMA(df['close'], order=(1, 1, 1)).fit()
    return model, df

def evaluate_forecast(df, forecast_data):
    print("Evaluation Metrics:")
    print("Mean Forecast Value:", forecast_data['forecast'].mean())


def plot_stock_prices(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['close'], label='Stock Prices')
    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Fetch data function remains unchanged
def fetch_data():
    # Simulated stock data
    dates = pd.date_range(start="2024-10-18", periods=100, freq='B')  # Business days
    close_prices = np.random.uniform(378, 455, size=100).round(2)
    df = pd.DataFrame({'date': dates, 'close': close_prices}).set_index('date')
    return df

# Main function remains largely unchanged
def main():
    print("Fetching stock data...")
    df = fetch_data()

    print("Plotting stock prices...")
    plot_stock_prices(df)

    print("Forecasting using ARIMA...")
    model, forecast_df = train_arima_model(df)

    # Predict for the next week (5 business days)
    print("Predicting the next 5 business days...")
    forecast_steps = 5
    forecast_result = model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Append forecast to DataFrame
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    forecast_data = pd.DataFrame({
        'forecast': forecast_mean.values,
        'lower_bound': conf_int.iloc[:, 0],
        'upper_bound': conf_int.iloc[:, 1]
    }, index=future_dates)

    # Plot observed data, forecast, and confidence intervals
    print("Plotting forecast with confidence intervals...")
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Observed')
    plt.plot(forecast_data.index, forecast_data['forecast'], label='Forecast', color='orange')
    plt.fill_between(forecast_data.index, forecast_data['lower_bound'], forecast_data['upper_bound'], 
                     color='gray', alpha=0.3, label='Confidence Interval')
    plt.title('ARIMA Forecast with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    print("Evaluating forecast performance...")
    evaluate_forecast(df, forecast_data)

    print("Additional Insights:")
    print("1. Forecasted next 5 business days values:\n", forecast_mean.values)
    print("2. Confidence intervals for the predictions:\n", conf_int)

if __name__ == "__main__":
    main()
