from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI()

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    initialBalance: int
    startYear: int
    endYear: int
    assets: list

@app.get("/msp/stocks/{symbol}")
def retrieve_stock_name(symbol: str):
    try:
      data = yf.Ticker(symbol).info
      companyName = data['longName']
      return ({'name': companyName})
    except:
       raise HTTPException(status_code=404, detail="Symbol data not found")

@app.post("/msp/stats")
async def portfolio_stats(portfolio: Item):
    stocks = []
    weights = []
    for asset in portfolio.assets:
        stocks.append(asset['symbol'])
        weights.append(int(asset['weight'])/100)
    
    final_balance = compute_final_balance(portfolio.initialBalance, stocks, weights, portfolio.startYear, portfolio.endYear)
    mdd = compute_mdd(stocks, weights, portfolio.startYear, portfolio.endYear)
    port_cagr = compute_cagr(stocks, weights, portfolio.startYear, portfolio.endYear)
    port_vol = compute_portfolio_volatility(stocks, weights, portfolio.startYear, portfolio.endYear)
    sharpe = compute_sharpe_ratio(stocks, weights, portfolio.startYear, portfolio.endYear, risk_free=0.05)

    dd_data = drawdown_data(stocks, weights, portfolio.startYear, portfolio.endYear)
    growth_data = port_growth_data(stocks, weights, portfolio.startYear, portfolio.endYear)

    return {'final_bal': final_balance, 'mdd': mdd, 'port_cagr': port_cagr, 'port_vol': port_vol, 'sharpe': sharpe, 'dd_data': dd_data, 'growth_data': growth_data}


def compute_mdd(stocks, weights, start_year:int, end_year:int):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    weights_np = np.array(weights)

    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = yf.Ticker(stock).history(period='max', start=start_date, end=end_date)['Close']

    # compute daily returns
    daily_returns = df.pct_change().dropna()
    # compute portfolio daily returns
    daily_returns['port'] = daily_returns.dot(weights_np)

    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns.fillna(1, inplace=True)

    portfolio_cumulative_returns = cumulative_returns['port']

    previous_peaks = portfolio_cumulative_returns.cummax()
    drawdown = (portfolio_cumulative_returns - previous_peaks) / previous_peaks
    mdd = drawdown.min()
    return mdd

def drawdown_data(stocks, weights, start_year:int, end_year:int):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    weights_np = np.array(weights)

    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = yf.Ticker(stock).history(period='max', start=start_date, end=end_date)['Close']

    # compute daily returns
    daily_returns = df.pct_change().dropna()
    # compute portfolio daily returns
    daily_returns['port'] = daily_returns.dot(weights_np)

    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns.fillna(1, inplace=True)

    portfolio_cumulative_returns = cumulative_returns['port']

    previous_peaks = portfolio_cumulative_returns.cummax()
    drawdown = (portfolio_cumulative_returns - previous_peaks) / previous_peaks
    return drawdown.to_json()

def port_growth_data(stocks, weights, start_year:int, end_year:int):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    weights_np = np.array(weights)

    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = yf.Ticker(stock).history(period='max', start=start_date, end=end_date)['Close']

    # compute daily returns
    daily_returns = df.pct_change().dropna()
    # compute portfolio daily returns
    daily_returns['port'] = daily_returns.dot(weights_np)

    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns.fillna(1, inplace=True)

    portfolio_cumulative_returns = cumulative_returns['port']
    return portfolio_cumulative_returns.to_json()

def compute_cagr(stocks, weights, start_year:int, end_year:int):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    weights_np = np.array(weights)

    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = yf.Ticker(stock).history(period='max', start=start_date, end=end_date)['Close']

    daily_returns = df.pct_change()

    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns.fillna(1,inplace=True)
    cagr = cumulative_returns**(252/len(cumulative_returns.index)) - 1

    portfolio_cagr = np.dot(cagr.iloc[-1,:],weights_np)
    return portfolio_cagr

def compute_portfolio_volatility(stocks, weights, start_year:int, end_year:int):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    weights_np = np.array(weights)

    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = yf.Ticker(stock).history(period='max', start=start_date, end=end_date)['Close']

    daily_returns = df.pct_change()

    # From daily returns to portfolio volatility

    cov_matrix = daily_returns.cov() * 252

    # Calculate the portfolio variance

    portfolio_vol = np.sqrt(np.dot(weights_np.T, np.dot(cov_matrix, weights_np)))
    return portfolio_vol

def compute_sharpe_ratio(stocks, weights, start_year:int, end_year:int, risk_free = 0.0):
    port_cagr = compute_cagr(stocks, weights, start_year, end_year)
    port_vol = compute_portfolio_volatility(stocks, weights, start_year, end_year)
    return (port_cagr - risk_free) / port_vol

def compute_final_balance(balance, stocks, weights, start_year:int, end_year:int, risk_free = 0.0):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    weights_np = np.array(weights)

    df = pd.DataFrame()

    for stock in stocks:
        df[stock] = yf.Ticker(stock).history(period='max', start=start_date, end=end_date)['Close']

    daily_returns = df.pct_change().dropna()
    daily_returns['port'] = daily_returns.dot(weights_np)

    cumulative_returns = (1 + daily_returns).cumprod()
    cumulative_returns.fillna(1, inplace=True)

    portfolio_cumulative_returns = cumulative_returns['port']
    return balance * portfolio_cumulative_returns.iloc[-1]
    