import yfinance as yf
import pandas as pd

def ticker_detail(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
   
    data = {
        "Current Price $": [stock_info.get("currentPrice")],
        "Open Price $": [stock_info.get("open")],
        "Bid Price $": [stock_info.get("bid")],
        "Ask Price $": [stock_info.get("ask")],
        "Previous Close $": [stock_info.get("previousClose")],
        "Beta": [stock_info.get("beta")],
        "Day High $": [stock_info.get("dayHigh")],
        "Day Low $": [stock_info.get("dayLow")],
        "Market Cap $": [stock_info.get("marketCap")],
        "PE Ratio (TTM)": [stock_info.get("trailingPE")],
        "EPS (TTM) $": [stock_info.get("trailingEps")],
        "Volume": [stock_info.get("volume")],
        "Industry": [stock_info.get("sector")]
}

    #df = pd.DataFrame(data).T
    return data

ticker_detail('AAPL')