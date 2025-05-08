import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🤖 Crypto AI Signals: Model-Driven Entries with TP/SL")

# Config
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = {'1h': '1h', '4h': '4h', '1d': '1d'}

# Sidebar
symbol = st.sidebar.selectbox("Select Symbol", SYMBOLS)
timeframe = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))
tp_multiplier = st.sidebar.slider("TP Multiplier", 1.0, 5.0, 2.0)
sl_multiplier = st.sidebar.slider("SL Multiplier", 0.5, 3.0, 1.5)

# Fetch historical data
def get_klines(symbol, interval, limit=300):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Add indicators
def apply_indicators(df):
    df['EMA_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['ATR'] = atr.average_true_range()
    df.dropna(inplace=True)
    return df

# Prepare data for model
def prepare_sequences(df, seq_len, feature_cols):
    X = []
    for i in range(len(df) - seq_len):
        X.append(df[feature_cols].iloc[i:i+seq_len].values)
    return np.array(X)

# Build simple LSTM model
def build_model(input_shape):
    model = tf.keras.Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Generate AI signal
def ai_signal_prediction(model, X):
    pred = model.predict(X, verbose=0)
    return 'Buy' if pred[-1][0] > 0.6 else 'Sell' if pred[-1][0] < 0.4 else 'Neutral'

# Main
with st.spinner("Fetching and processing data..."):
    df = get_klines(symbol, TIMEFRAMES[timeframe])
    df = apply_indicators(df)

# Feature columns
feature_cols = ['close', 'volume', 'EMA_9', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
seq_len = 20
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols, index=df.index)
X_seq = prepare_sequences(df_scaled, seq_len, feature_cols)

# Build & load model (note: in real use, you'd load a pre-trained model)
model = build_model((seq_len, len(feature_cols)))

# For demonstration, simulate quick training (only for demo, not for real trading)
y_dummy = np.random.randint(0, 2, size=(X_seq.shape[0],))
model.fit(X_seq, y_dummy, epochs=1, batch_size=16, verbose=0)

# Predict signal
signal = ai_signal_prediction(model, X_seq)

# Get TP/SL
def calculate_tp_sl(row, signal):
    entry = row['close']
    atr = row['ATR']
    if signal == 'Buy':
        tp = entry + tp_multiplier * atr
        sl = entry - sl_multiplier * atr
    elif signal == 'Sell':
        tp = entry - tp_multiplier * atr
        sl = entry + sl_multiplier * atr
    else:
        tp = sl = None
    return round(tp, 4) if tp else None, round(sl, 4) if sl else None

# Plot chart
st.subheader(f"{symbol} {timeframe} Chart & AI Signal")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='red')))
st.plotly_chart(fig, use_container_width=True)

# Show signal
if not df.empty:
    last_row = df.iloc[-1]
    tp, sl = calculate_tp_sl(last_row, signal)
    st.subheader("🤖 AI Model Signal")
    st.write(f"Signal: **{signal}**")
    if signal in ['Buy', 'Sell']:
        st.write(f"Best Entry: {last_row['close']:.4f}")
        st.write(f"Take Profit (TP): {tp}")
        st.write(f"Stop Loss (SL): {sl}")
    else:
        st.write("No actionable signal from the model at the moment.")
