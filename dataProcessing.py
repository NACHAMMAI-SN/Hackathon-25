import pandas as pd

# Load the data
df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
print("✅ Data loaded successfully!")

# Step 1: Handle missing values in raw data (before feature engineering)
df = df.ffill().bfill()  # Fill missing values forward, then backward to cover all cases

# Step 2: Feature Engineering
df["daily_return"] = df["close"].pct_change()
df["rolling_avg"] = df["close"].rolling(window=7, min_periods=1).mean()  # Ensure no NaN at the start
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# Step 3: Handle missing values in derived features
df["daily_return"] = df["daily_return"].fillna(0)  # Replace NaN in daily_return with 0
df["rolling_avg"] = df["rolling_avg"].fillna(method='bfill')  # Backfill rolling_avg instead of using a static mean

# Step 4: Final check for any remaining NaN values (paka confirmation)
if df.isna().sum().sum() == 0:
    print(" No missing values left after preprocessing!")
else:
    print("⚠️ Warning: Some NaNs still exist!")

# Save cleaned data
df.to_csv("processed_stock_data.csv")
print(" Cleaned data saved to processed_stock_data.csv 🎯")
