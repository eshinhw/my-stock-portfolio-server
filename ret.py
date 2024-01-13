import yfinance as yf
import pandas as pd

# def compute_relative_strength():
# Define the ticker symbol
ticker_symbol = 'spy'

# Define the start and end dates for the historical data
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(months=6)

# Fetch historical data from Yahoo Finance
spy_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Calculate the percentage change over the 6-month period
six_month_return = ((spy_data['Adj Close'].iloc[-1] / spy_data['Adj Close'].iloc[0]) - 1) * 100

print(f"6-Month Return for {ticker_symbol}: {six_month_return:.2f}%")

