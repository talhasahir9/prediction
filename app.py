import streamlit as st import pandas as pd import numpy as np import requests from ta.trend import EMAIndicator, MACD from ta.momentum import RSIIndicator from ta.volatility import BollingerBands, AverageTrueRange from sklearn.preprocessing import MinMaxScaler import tensorflow as tf from tensorflow.keras.layers import LSTM, Dense, Dropout import xgboost as xgb import plotly.graph_objects as go

st.set_page_config(layout="wide") st.title("🔗 Crypto Dashboard with LSTM + XGBoost Ensemble and Advanced Backtesting")

Config

SYMBOLS = ['BTCUSDT', 'ETHUSDT'] TIMEFRAMES = {'1h': '1h', '4h': '4h', '1d': '1d'}

Sidebar

symbol = st.sidebar.selectbox("Select Symbol", SYMBOLS) timeframe = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))

Fetch historical data

def get_klines(symbol, interval, limit=300): url = "https://fapi.binance.com/fapi/v1/klines" params = {"symbol": symbol, "interval": interval, "limit": limit} data = requests.get(url, params=params).json() df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore']) df['time'] = pd.to_datetime(df['time'], unit='ms') df.set_index('time', inplace=True) df = df[['open', 'high', 'low', 'close', 'volume']].astype(float) return df

Add indicators

def apply_indicators(df): df['EMA_9'] = EMAIndicator(df['close'], window=9).ema_indicator() df['EMA_21'] = EMAIndicator(df['close'], window=21).ema_indicator() df['RSI'] = RSIIndicator(df['close']).rsi() macd = MACD(df['close']) df['MACD'] = macd.macd() df['MACD_Signal'] = macd.macd_signal() bb = BollingerBands(df['close']) df['BB_Upper'] = bb.bollinger_hband() df['BB_Lower'] = bb.bollinger_lband() df.dropna(inplace=True) return df

Create sequences for LSTM

def create_sequences(data, seq_len, feature_cols, target_col): X, y = [], [] for i in range(len(data) - seq_len): X.append(data[feature_cols].iloc[i:i+seq_len].values) y.append(data[target_col].iloc[i+seq_len]) return np.array(X), np.array(y)

Train LSTM

@st.cache_resource def train_lstm(X_train, y_train): model = tf.keras.Sequential([ LSTM(64, return_sequences=True, input_shape=X_train.shape[1:]), LSTM(32), Dropout(0.2), Dense(1) ]) model.compile(optimizer='adam', loss='mse') model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0) return model

Train XGBoost

@st.cache_resource def train_xgboost(X_train, y_train): model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05) model.fit(X_train, y_train) return model

Backtest function

def backtest(df, lstm_model, xgb_model, scaler, feature_cols, seq_len): balance = 1000  # Starting balance positions = [] returns = [] predictions, actuals = [], []

for i in range(seq_len, len(df)):
    seq_input = scaler.transform(df[feature_cols].iloc[i-seq_len:i])
    lstm_pred_scaled = lstm_model.predict(np.expand_dims(seq_input, axis=0), verbose=0)[0][0]
    xgb_pred_scaled = xgb_model.predict([scaler.transform(df[feature_cols].iloc[[i]])[0]])[0]

    latest_features = scaler.transform(df[feature_cols].iloc[[i]])[0]
    latest_features_copy = latest_features.copy()
    latest_features_copy[feature_cols.index('close')] = lstm_pred_scaled
    lstm_pred = scaler.inverse_transform([latest_features_copy])[0][0]
    latest_features_copy[feature_cols.index('close')] = xgb_pred_scaled
    xgb_pred = scaler.inverse_transform([latest_features_copy])[0][0]
    ensemble_pred = (lstm_pred + xgb_pred) / 2

    actual_close = df['close'].iloc[i]
    prev_close = df['close'].iloc[i-1]

    # Simulate trade: Buy if prediction > prev close, Sell if < prev close
    if ensemble_pred > prev_close:
        profit = (actual_close - prev_close) / prev_close
        positions.append(1)
    else:
        profit = (prev_close - actual_close) / prev_close
        positions.append(-1)
    returns.append(profit)

    predictions.append(ensemble_pred)
    actuals.append(actual_close)

returns = np.array(returns)
win_rate = np.sum(returns > 0) / len(returns) * 100
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
cumulative = np.cumprod(1 + returns)
max_drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative)

metrics = {
    'Win Rate (%)': round(win_rate, 2),
    'Sharpe Ratio': round(sharpe_ratio, 2),
    'Max Drawdown': round(max_drawdown * 100, 2)
}

return predictions, actuals, cumulative, metrics

Main

with st.spinner("Fetching and processing data..."): df = get_klines(symbol, TIMEFRAMES[timeframe]) df = apply_indicators(df)

st.subheader(f"{symbol} {timeframe} Chart") fig = go.Figure() fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])) fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='blue'))) fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='red'))) st.plotly_chart(fig, use_container_width=True)

Prepare data

feature_cols = ['close', 'volume', 'EMA_9', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower'] target_col = 'close' seq_len = 20 scaler = MinMaxScaler() df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols, index=df.index) X_lstm, y_lstm = create_sequences(df_scaled, seq_len, feature_cols, target_col) X_xgb = df_scaled[feature_cols].iloc[seq_len:].values y_xgb = df[target_col].iloc[seq_len:].values

Train models

lstm_model = train_lstm(X_lstm, y_lstm) xgb_model = train_xgboost(X_xgb, y_xgb)

Predictions

latest_seq = np.expand_dims(df_scaled[feature_cols].iloc[-seq_len:].values, axis=0) lstm_pred_scaled = lstm_model.predict(latest_seq, verbose=0)[0][0] xgb_pred_scaled = xgb_model.predict([df_scaled[feature_cols].iloc[-1].values])[0] latest_features = df_scaled.iloc[-1].copy() latest_features['close'] = lstm_pred_scaled lstm_pred = scaler.inverse_transform([latest_features])[0][0] latest_features['close'] = xgb_pred_scaled xgb_pred = scaler.inverse_transform([latest_features])[0][0] ensemble_pred = (lstm_pred + xgb_pred) / 2

st.subheader("📈 Predictions") st.write(f"LSTM Prediction: {lstm_pred:.4f}") st.write(f"XGBoost Prediction: {xgb_pred:.4f}") st.success(f"Ensemble Prediction: {ensemble_pred:.4f}")

Backtesting

st.subheader("🔄 Advanced Backtesting") with st.spinner("Running backtest..."): preds, actuals, cumulative, metrics = backtest(df, lstm_model, xgb_model, scaler, feature_cols, seq_len) backtest_df = pd.DataFrame({'Prediction': preds, 'Actual': actuals}, index=df.index[seq_len:]) fig_bt = go.Figure() fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual'], name='Actual')) fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Prediction'], name='Prediction')) st.plotly_chart(fig_bt, use_container_width=True)

# Metrics
st.subheader("Performance Metrics")
for k, v in metrics.items():
    st.write(f"{k}: {v}")

# Cumulative returns plot
st.subheader("Cumulative Returns")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=backtest_df.index, y=cumulative, name='Cumulative Return'))
st.plotly_chart(fig_cum, use_container_width=True)
