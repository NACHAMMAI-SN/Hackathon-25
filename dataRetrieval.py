import requests
import pandas as pd

def fetch_data(api_url, params):
    """
    Fetch data from a REST API.
    """
    print("Fetching data from API...")
    response = requests.get(api_url, params=params)
    print(f"API Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("Data fetched successfully!")
        return data
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Text: {response.text}")  # Print the response text for debugging
        return None

# Example: Fetch stock data from Alpha Vantage
api_key = "AW084LNN3NVBC1AO"  # Replace with your API key
symbol = "TSLA"  # Apple Inc.
api_url = "https://www.alphavantage.co/query"
params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "apikey": api_key,
    "outputsize": "full"  # Get full historical data
}

# Fetch data
print("Starting data retrieval...")
data = fetch_data(api_url, params)

if data:
    # Convert JSON to DataFrame
    print("Converting JSON to DataFrame...")
    time_series = data.get("Time Series (Daily)")
    if time_series:
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df.columns = ["open", "high", "low", "close", "volume"]
        
        # Save data to a CSV file
        print("Saving data to CSV...")
        df.to_csv("stock_data.csv")
        print("Data saved to stock_data.csv")
    else:
        print("Error: 'Time Series (Daily)' key not found in API response.")
else:
    print("Failed to fetch data.")