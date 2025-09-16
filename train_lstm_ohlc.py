# train_lstm_ohlc.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import joblib
import os

def prepare_ohlc_dataset(df, window_size=10):
    df = df.sort_values("Date")
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(scaled_features) - window_size):
        X.append(scaled_features[i:i+window_size])
        y.append(scaled_features[i+window_size][:4])  # predict next day's OHLC only

    return np.array(X), np.array(y), scaler

def train_lstm_ohlc(csv_path, symbol, window_size=10, epochs=100):
    df = pd.read_csv(csv_path)
    X, y, scaler = prepare_ohlc_dataset(df, window_size)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4))  # Output: [Open, High, Low, Close]

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1,
              callbacks=[EarlyStopping(patience=10)], verbose=1)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol}_lstm_ohlc.h5")
    joblib.dump(scaler, f"models/{symbol}_ohlc_scaler.pkl")
    joblib.dump(mse, f"models/{symbol}_ohlc_mse.pkl")

    print(f"✅ Model trained and saved as models/{symbol}_lstm_ohlc.h5")
    print(f"📊 MSE: {round(mse, 5)}")

if __name__ == "__main__":
    train_lstm_ohlc("data/AAPL_historical.csv", "AAPL")
