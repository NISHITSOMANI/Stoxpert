
import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = '44011697b1a84be594c5852f6e0d111b'  # Replace with your key

def fetch_live_data(symbol):
    url = f'https://api.twelvedata.com/quote?symbol={symbol}&apikey={API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()

        if 'status' in data and data['status'] == 'error':
            print("Live data API error:", data['message'])
            return None

        price = float(data.get("price") or data.get("close"))

        return {
            "symbol": data["symbol"],
            "price": price,
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low": float(data["low"]),
            "prev_close": float(data["previous_close"])
        }
    except Exception as e:
        print("Live data fetch error:", e)
        return None


def fetch_historical_data(symbol, interval='1day', outputsize=500):
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()
        values = data['values']
        df = pd.DataFrame(values)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.astype({
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float'
        })
        return df[::-1]  # Oldest first
    except Exception as e:
        print("Historical data fetch error:", e)
        return None


def save_to_csv(df, symbol):
    folder = 'data'
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{symbol.upper()}_historical.csv"
    df.to_csv(filename, index=False)
    return filename


def predict_next_5_day_average(df):
    try:
        if df is None or len(df) < 5:
            return None
        avg = df['Close'].tail(5).mean()
        return [round(avg, 2)] * 5
    except Exception as e:
        print("Prediction error:", e)
        return None
