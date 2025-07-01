from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from data_fetcher import fetch_live_data, fetch_historical_data, save_to_csv
import plotly.graph_objs as go
import plotly
import json

app = Flask(__name__)

def create_candlestick_chart(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # convert to string

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'].tolist(),  # Convert to list
        open=df['Open'].tolist(),
        high=df['High'].tolist(),
        low=df['Low'].tolist(),
        close=df['Close'].tolist()
    )])

    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
from sklearn.metrics import mean_squared_error

def train_lstm(csv_path, symbol, window_size=10, epochs=100):
    df = pd.read_csv(csv_path).sort_values("Date")
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size])

    X, y = np.array(X), np.array(y)
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1,
              callbacks=[EarlyStopping(patience=10)], verbose=0)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol}_lstm.h5")
    joblib.dump(scaler, f"models/{symbol}_scaler.pkl")
    joblib.dump(mse, f"models/{symbol}_lstm_mse.pkl")  # save accuracy

    return round(mse, 4)


def predict_next_day_lstm(csv_path, symbol, window_size=10):
    try:
        model_path = f"models/{symbol}_lstm.h5"
        scaler_path = f"models/{symbol}_scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"⚠️ Model or scaler missing. Auto-training LSTM for {symbol}")
            train_lstm(csv_path, symbol)

        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        df = pd.read_csv(csv_path).sort_values("Date")
        data = df["Close"].values.reshape(-1, 1)
        scaled = scaler.transform(data)
        last_sequence = scaled[-window_size:].reshape(1, window_size, 1)

        prediction_scaled = model.predict(last_sequence)[0][0]
        predicted_price = scaler.inverse_transform([[prediction_scaled]])[0][0]

        try:
            mse = joblib.load(f"models/{symbol}_lstm_mse.pkl")
        except:
            mse = None
        return round(predicted_price, 2), round(mse, 4) if mse else None


    except Exception as e:  
        print("❌ LSTM Prediction Error:", e)
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_data = None
    candlestick = None
    ai_forecast = None
    error = None
    model_accuracy= None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper().strip()
        stock_data = fetch_live_data(symbol)
        historical_df = fetch_historical_data(symbol, interval='1day', outputsize=300)

        if stock_data and historical_df is not None:
            csv_path = save_to_csv(historical_df, symbol)
            candlestick = create_candlestick_chart(historical_df)
            ai_forecast,model_accuracy = predict_next_day_lstm(csv_path, symbol)
        else:
            error = "Error fetching data. Please try again later or check symbol."

    return render_template(
        'index.html',
        stock_data=stock_data,
        candlestick=candlestick,
        ai_forecast=ai_forecast,
        model_accuracy=model_accuracy,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
