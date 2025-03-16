import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting App", page_icon="📈", layout="wide")

# Title of the app
st.title("📈 Time Series Forecasting App")
st.write("This app forecasts time series data using ARIMA and provides insights through EDA and visualization.")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Initialize the `data` variable
data = None

# Milestone 1: Data Retrieval
st.sidebar.subheader("Milestone 1: Data Retrieval")
api_url = st.sidebar.text_input("Enter API URL", "https://www.alphavantage.co/query")
api_key = st.sidebar.text_input("Enter API Key", "AW084LNN3NVBC1AO")
symbol = st.sidebar.text_input("Enter Symbol", "AAPL")
params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "apikey": api_key,
    "outputsize": "full"
}

def fetch_data(api_url, params):
    """
    Fetch data from a REST API.
    """
    st.write("Fetching data from API...")
    response = requests.get(api_url, params=params)
    st.write(f"API Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        st.success("Data fetched successfully!")
        return data
    else:
        st.error(f"Error: {response.status_code}")
        st.error(f"Response Text: {response.text}")
        return None

if st.sidebar.button("Fetch Data"):
    raw_data = fetch_data(api_url, params)  # Keep raw_data as a dictionary
    if raw_data and "Time Series (Daily)" in raw_data:
        time_series = raw_data["Time Series (Daily)"]
        data = pd.DataFrame(time_series).T  # Convert to DataFrame
        data.index = pd.to_datetime(data.index)
        data.columns = ["open", "high", "low", "close", "volume"]
        data = data.astype(float)  # Convert to numeric
        data.to_csv("stock_data.csv")
        st.success("Data saved to stock_data.csv")
    else:
        st.error("Error: Invalid API response or missing 'Time Series (Daily)' key.")

# Milestone 2: Data Cleaning and Preprocessing
st.sidebar.subheader("Milestone 2: Data Cleaning and Preprocessing")
uploaded_file = st.sidebar.file_uploader("Upload your time series data (CSV)", type="csv")

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)

    # Check if the data has a date column
    date_column = st.sidebar.selectbox("Select the date column", data.columns)
    if date_column:
        # Convert to datetime and set as index
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)

        # Ensure the index is monotonic and has a frequency
        data = data.sort_index()  # Sort by date
        data = data.asfreq('D')   # Set frequency to daily ('D')

        st.write("### Uploaded Data")
        st.write(data.head())

        # Handle missing values
        st.write("### Handling Missing Values")
        data = data.ffill().bfill()
        st.write("Missing values after handling:")
        st.write(data.isnull().sum())

        # Feature Engineering
        st.write("### Feature Engineering")
        if "close" in data.columns:
            data["daily_return"] = data["close"].pct_change()
            data["rolling_avg"] = data["close"].rolling(window=7, min_periods=1).mean()
        data["day_of_week"] = data.index.dayofweek
        data["month"] = data.index.month

        # Save cleaned data
        data.to_csv("processed_stock_data.csv")
        st.success("Cleaned data saved to processed_stock_data.csv")

# Milestone 3: Exploratory Data Analysis (EDA)
st.sidebar.subheader("Milestone 3: Exploratory Data Analysis (EDA)")
if st.sidebar.button("Perform EDA"):
    if data is not None:
        st.write("### Basic EDA")
        st.write("#### Time Series Plot")
        plt.figure(figsize=(12, 6))
        if "close" in data.columns:
            plt.plot(data.index, data["close"], label="Close Price")
        else:
            plt.plot(data.index, data.iloc[:, 0], label="Time Series Data")
        plt.title("Time Series Plot")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        st.pyplot(plt)

        st.write("#### Trend and Seasonality")
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        if "close" in data.columns:
            sns.lineplot(x=data.index, y=data["close"], label="Close Price")
        else:
            sns.lineplot(x=data.index, y=data.iloc[:, 0], label="Time Series Data")
        plt.title("Trend and Seasonality")
        plt.xlabel("Date")
        plt.ylabel("Value")
        st.pyplot(plt)

        if "daily_return" in data.columns:
            st.write("#### Heatmap of Daily Returns")
            pivot_table = data.pivot_table(index=data.index.month, columns=data.index.day, values="daily_return")
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_table, cmap="viridis")
            plt.title("Heatmap of Daily Returns")
            st.pyplot(plt)
    else:
        st.error("No data available. Please upload a CSV file or fetch data from the API.")

# Milestone 4: Model Building and Evaluation
st.sidebar.subheader("Milestone 4: Model Building and Evaluation")
if data is not None:
    target_column = st.sidebar.selectbox("Select the target column (to forecast)", data.columns, key="target_column")
    exogenous_columns = st.sidebar.multiselect("Select exogenous variables (optional)", data.columns, key="exogenous_columns")

    # Input fields for ARIMA parameters
    p = st.sidebar.number_input("AR Order (p)", min_value=0, max_value=5, value=1, key="p")
    d = st.sidebar.number_input("Difference Order (d)", min_value=0, max_value=2, value=1, key="d")
    q = st.sidebar.number_input("MA Order (q)", min_value=0, max_value=5, value=1, key="q")

    if st.sidebar.button("Train Model", key="train_model"):
        st.write("### Model Training and Evaluation")
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]

        # Check stationarity
        st.write("### Checking Stationarity")
        def check_stationarity(series):
            result = adfuller(series)
            st.write(f"ADF Statistic: {result[0]}")
            st.write(f"p-value: {result[1]}")
            if result[1] > 0.05:
                st.warning("Series is not stationary. Apply differencing or transformations.")
                return False
            else:
                st.success("Series is stationary.")
                return True

        is_stationary = check_stationarity(train[target_column])

        # Apply differencing if necessary
        if not is_stationary:
            st.write("### Applying Differencing")
            train_diff = train[target_column].diff().dropna()
            is_stationary = check_stationarity(train_diff)
            if is_stationary:
                train.loc[:, target_column] = train_diff
                st.success("Differencing applied successfully.")
            else:
                st.error("Differencing did not make the series stationary. Try other transformations.")

        # Plot ACF and PACF to determine AR and MA orders
        st.write("### ACF and PACF Plots")
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plot_acf(train[target_column], lags=20, ax=plt.gca())
        plt.title("Autocorrelation Function (ACF)")

        plt.subplot(2, 1, 2)
        plot_pacf(train[target_column], lags=20, ax=plt.gca())
        plt.title("Partial Autocorrelation Function (PACF)")

        st.pyplot(plt)

        # Prepare exogenous variables (if any)
        exog_train = train[exogenous_columns] if exogenous_columns else None
        exog_test = test[exogenous_columns] if exogenous_columns else None

        try:
            # Fit ARIMA model
            model = ARIMA(train[target_column], order=(p, d, q), exog=exog_train)
            model_fit = model.fit()

            # Store model_fit and test data in session state
            st.session_state.model_fit = model_fit
            st.session_state.test = test

            # Forecast on the test set
            if exogenous_columns:
                forecast_result = model_fit.get_forecast(steps=len(test), exog=exog_test)
            else:
                forecast_result = model_fit.get_forecast(steps=len(test))
            
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            # Align forecast with test data
            test_forecast = forecast_values[:len(test)]

            # Calculate evaluation metrics
            mae = mean_absolute_error(test[target_column], test_forecast)
            rmse = np.sqrt(mean_squared_error(test[target_column], test_forecast))
            mape = np.mean(np.abs((test[target_column] - test_forecast) / test[target_column])) * 100

            st.write("#### Model Evaluation Metrics")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAPE: {mape:.2f}%")

            # Plot actual vs forecasted values
            st.write("#### Actual vs Forecasted Values")
            plt.figure(figsize=(12, 6))
            plt.plot(test.index, test[target_column], label="Actual", color="blue")
            plt.plot(test.index, test_forecast, label="Forecast", color="red")
            plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3, label="Confidence Interval")
            plt.title("Actual vs Forecasted Values")
            plt.xlabel("Date")
            plt.ylabel(target_column)
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error training the model: {e}")
else:
    st.sidebar.warning("No data available. Please upload a CSV file or fetch data from the API.")

# Milestone 5: Forecasting
st.sidebar.subheader("Milestone 5: Forecasting")
if data is not None:
    if 'model_fit' in st.session_state:  # Check if model is trained
        forecast_horizon = st.sidebar.number_input("Forecast Horizon (number of steps)", min_value=1, max_value=100, value=7, key="forecast_horizon")

       # Inside the "Generate Forecast" section
if st.sidebar.button("Generate Forecast", key="generate_forecast"):
    st.write("### Future Forecast")
    last_row = st.session_state.test.iloc[-1]
    exog_future = pd.DataFrame([last_row[exogenous_columns]]) if exogenous_columns else None

    try:
        if exogenous_columns:
            future_forecast = st.session_state.model_fit.get_forecast(steps=forecast_horizon, exog=exog_future)
        else:
            future_forecast = st.session_state.model_fit.get_forecast(steps=forecast_horizon)
        
        future_values = future_forecast.predicted_mean
        future_conf_int = future_forecast.conf_int()

        # Reconstruct the forecasted close price if differencing was applied
        if d > 0:  # If differencing was applied during model training
            last_close_price = st.session_state.test[target_column].iloc[-1]  # Last actual close price
            future_values = future_values.cumsum() + last_close_price  # Reconstruct original scale
            future_conf_int = future_conf_int.cumsum() + last_close_price  # Reconstruct confidence intervals

        # Create a datetime index for the forecasted values
        last_date = st.session_state.test.index[-1]  # Last date in the test data
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        future_values.index = forecast_dates
        future_conf_int.index = forecast_dates

        # Convert forecasted values to a DataFrame for better display
        forecast_df = pd.DataFrame({
            "Date": future_values.index,
            "Forecasted Close Price": future_values,
            "Lower Confidence Interval": future_conf_int.iloc[:, 0],
            "Upper Confidence Interval": future_conf_int.iloc[:, 1]
        })

        st.write("#### Forecasted Close Prices:")
        st.dataframe(forecast_df)  # Display as a table

        # Plot future forecast
        plt.figure(figsize=(12, 6))
        plt.plot(st.session_state.test.index, st.session_state.test[target_column], label="Actual Close Price", color="blue")
        plt.plot(future_values.index, future_values, label="Forecasted Close Price", color="green")
        plt.fill_between(future_values.index, future_conf_int.iloc[:, 0], future_conf_int.iloc[:, 1], color="lightgreen", alpha=0.3, label="Confidence Interval")
        plt.title(f"Forecast for the Next {forecast_horizon} Steps")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error generating forecast: {e}")
else:
    st.sidebar.warning("No data available. Please upload a CSV file or fetch data from the API.")