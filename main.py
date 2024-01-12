from typing import Union

from fastapi import FastAPI, HTTPException
import yfinance as yf

app = FastAPI()

def fetch_stock_data(symbol, period: str = "1mo", start_date: str = "2020-01-01", end_date: str = "2024-01-01"):
    try:
        stock_data = yf.download(symbol, period=period, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_earnings_data(symbol):
    earnings_data = yf.get_earnings_calendar(symbol)
    return earnings_data
    
def compute_sma(data, periods):
    return data['Adj Close'].rolling(window=periods).mean()

@app.get("/sma/{symbol}&period={period}&start_date={start_date}&end_date={end_date}")
async def get_sma(symbol: str, period, start_date: str = '2020-01-01', end_date: str = '2024-01-01'):
    try:
        # Fetch historical stock data
        stock_data = fetch_stock_data(symbol, period, start_date, end_date)
        print(stock_data)

        # Fetch historical earnings data
        # earnings_data = fetch_earnings_data(symbol)

        # Compute SMAs for different periods
        stock_data['20SMA'] = compute_sma(stock_data, 20)
        stock_data['50SMA'] = compute_sma(stock_data, 50)
        stock_data['100SMA'] = compute_sma(stock_data, 100)
        stock_data['150SMA'] = compute_sma(stock_data, 150)
        stock_data['200SMA'] = compute_sma(stock_data, 200)

        # Compute Market Cap
        stock_data['MarketCap'] = stock_data['Adj Close'] * stock_data['Volume']

        # Compute Relative Strength against SPY
        stock_data['RelativeStrength'] = stock_data['Adj Close'] / stock_data['Close'].mean()

        # Check if current price is above 20 SMA
        stock_data['Above20SMA'] = stock_data['Adj Close'] > stock_data['20SMA']

        # Check if lower time SMAs are above higher time SMAs
        stock_data['20SMA_Above_50SMA'] = stock_data['20SMA'] > stock_data['50SMA']
        stock_data['50SMA_Above_100SMA'] = stock_data['50SMA'] > stock_data['100SMA']
        stock_data['100SMA_Above_150SMA'] = stock_data['100SMA'] > stock_data['150SMA']
        stock_data['150SMA_Above_200SMA'] = stock_data['150SMA'] > stock_data['200SMA']

        # Check if all smas are aligned uptrend
        stock_data['Uptrend_Signal'] = stock_data[['Above20SMA', '20SMA_Above_50SMA', '50SMA_Above_100SMA', '100SMA_Above_150SMA', '150SMA_Above_200SMA']].all(axis="columns")
        

        # Compute EPS growth
        # eps_growth = []
        # for index, row in stock_data.iterrows():
        #     try:
        #         earnings_info = earnings_data.loc[index]
        #         current_eps = earnings_info['epsEstimate']['raw']
        #         previous_eps = earnings_info['epsActual']['raw']
        #         growth = ((current_eps - previous_eps) / abs(previous_eps)) * 100
        #         eps_growth.append(growth)
        #     except KeyError:
        #         eps_growth.append(None)

        # stock_data['EPSGrowth'] = eps_growth

        # Return the computed data
        sma_data = stock_data[['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume', 'MarketCap', 'RelativeStrength', 'Above20SMA',
                               'Uptrend_Signal']].dropna()
        print(sma_data)
        return {"symbol": symbol, "data": sma_data.to_dict(orient='records')}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {symbol}: {e}")





# def retrieve_ticker_data(ticker: str, window_size: int = 20, period: str = '1mo'):
#     stock_data = get_stock_data(ticker, period=period)

#     if stock_data.empty:
#         raise HTTPException(status_code=404, detail=f"No data available for {ticker}")

#     stock_data['MA'] = stock_data['Close'].rolling(window=window_size).mean()
    
#     result = stock_data[['Close', 'MA']].dropna().to_dict(orient='records')

#     return result