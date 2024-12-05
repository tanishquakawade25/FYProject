import yfinance as yf

# Download TCS data
tcs = yf.download("TCS.NS", start="2024-01-01", end="2024-12-01")

# Check if data is empty
if tcs.empty:
    print("No data found for TCS.NS. Verify symbol and provider.")
else:
    print(tcs.head())
