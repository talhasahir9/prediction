import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout
import plotly.graph_objects as go
import websocket
import json
import threading
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import pywt
from datetime import datetime

# Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
TIMEFRAMES = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '1d': '1d'}
BASE_URL = "https://fapi.binance.com"
LIMIT = 500
SEQUENCE_LENGTH = 20
FORECAST_HORIZON = 5

st.set_page_config(layout="wide")
st.title("📊 Cutting-Edge Crypto Dashboard with Transformer Predictions")

# API Keys (replace with your actual keys)
CRYPTOPANIC_API_KEY = "YOUR_CRYPTOPANIC_API_KEY_HERE"
X_API_KEY = "YOUR_X_API_KEY_HERE"
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY_HERE"

# Real-Time Data Storage
realtime_data = {}
order_book_data = {}

# WebSocket for Real-Time Kline and Order Book Data
def on_message(ws, message):
    data = json.loads(message)
    if 'k' in data:
        kline = data['k']
        symbol = kline['s']
        df = pd.DataFrame([{
            'time': pd.to_datetime(kline['t'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v'])
        }])
        df.set_index('time', inplace=True)
        if symbol in realtime_data:
            realtime_data[symbol] = pd.concat([realtime_data[symbol], df]).tail(LIMIT).drop_duplicates()
        else:
            realtime_data[symbol] = df
    elif 'bids' in data and 'asks' in data:
        symbol = data['s']
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)
        bid_volume = np.sum(bids[:, 1])  # Sum of bid quantities
        ask_volume = np.sum(asks[:, 1])  # Sum of ask quantities
        order_book_data[symbol] = {'bid_volume': bid_volume, 'ask_volume': ask_volume, 'time': pd.to_datetime(data['E'], unit='ms')}

def on_error(ws, error):
    st.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket closed")

def start_websocket(symbol, interval):
    ws_url = f"wss://fstream.binance.com/stream?streams={symbol.lower()}@kline_{interval}/{symbol.lower()}@depth10@100ms"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

# Data Fetching with Caching
@st.cache_data
def get_klines(symbol, interval, limit=LIMIT):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Fetch Order Book Snapshot
@st.cache_data(ttl=300)
def get_order_book(symbol):
    url = f"{BASE_URL}/fapi/v1/depth"
    params = {"symbol": symbol, "limit": 10}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    r = requests.get(url, params=params, headers=headers)
    data = r.json()
    bids = np.array(data['bids'], dtype=float)
    asks = np.array(data['asks'], dtype=float)
    return {'bid_volume': np.sum(bids[:, 1]), 'ask_volume': np.sum(asks[:, 1])}

# Fetch X Sentiment with Topic Filtering
@st.cache_data(ttl=3600)
def get_x_sentiment(symbol):
    try:
        url = "https://api.x.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {X_API_KEY}"}
        query = f"{symbol[:3]} crypto (price OR market OR sentiment) -from:bot lang:en"
        params = {"query": query, "max_results": 10}
        response = requests.get(url, headers=headers, params=params)
        tweets = response.json().get('data', [])
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(tweet['text'])['compound'] for tweet in tweets]
        return sum(scores) / len(scores) if scores else 0
    except:
        return 0

# Apply Technical Indicators and Feature Engineering
def apply_indicators(df):
    if len(df) < 50:
        st.warning("Insufficient data for indicator calculation.")
        return df
    df['EMA_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['ATR'] = atr.average_true_range()
    df['min'] = df['low'].rolling(window=30).min()
    df['max'] = df['high'].rolling(window=30).max()
    
    # Feature Engineering
    df['pct_change'] = df['close'].pct_change()
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['rolling_mean'] = df['close'].rolling(window=10).mean()
    df['rolling_std'] = df['close'].rolling(window=10).std()
    df['volatility'] = df['pct_change'].rolling(window=10).std()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Wavelet Transform
    coeffs = pywt.wavedec(df['close'], 'db1', level=2)
    df['wavelet_trend'] = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], 'db1')[:len(df)]
    
    # Cross-Asset Correlation
    if 'BTCUSDT' in SYMBOLS and selected_symbol != 'BTCUSDT':
        btc_df = get_klines('BTCUSDT', TIMEFRAMES[list(TIMEFRAMES.keys())[0]])
        df['btc_correlation'] = df['close'].rolling(window=20).corr(btc_df['close'])
    else:
        df['btc_correlation'] = 1.0
    
    # Market Regime
    df['volatility_regime'] = pd.qcut(df['volatility'], q=3, labels=[0, 1, 2], duplicates='drop').astype(float)
    
    # Order Flow Imbalance
    if selected_symbol in order_book_data:
        df['order_flow_imbalance'] = (order_book_data[selected_symbol]['bid_volume'] - order_book_data[selected_symbol]['ask_volume']) / (order_book_data[selected_symbol]['bid_volume'] + order_book_data[selected_symbol]['ask_volume'])
    else:
        df['order_flow_imbalance'] = 0.0
    
    return df

# Prepare Transformer Sequences
def create_sequences(data, seq_length, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - seq_length - FORECAST_HORIZON):
        X.append(data[feature_cols].iloc[i:i+seq_length].values)
        y.append(data[target_col].iloc[i+seq_length+FORECAST_HORIZON-1])
    return np.array(X), np.array(y)

# Transformer Model
def build_transformer_model(seq_length, feature_dim, num_heads, num_layers, d_model, dropout_rate):
    inputs = tf.keras.Input(shape=(seq_length, feature_dim))
    x = Dense(d_model)(inputs)
    for _ in range(num_layers):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = Dense(d_model, activation='relu')(x)
        ffn = Dense(d_model)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    return tf.keras.Model(inputs, outputs)

# Objective Function for Bayesian Optimization
def objective(trial, df, feature_cols, target_col='close'):
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    d_model = trial.suggest_int('d_model', 32, 128, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols + [target_col], index=df.index)
    
    X, y = create_sequences(scaled_df, SEQUENCE_LENGTH, feature_cols, target_col)
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    model = build_transformer_model(SEQUENCE_LENGTH, len(feature_cols), num_heads, num_layers, d_model, dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
    
    y_pred = model.predict(X_test, verbose=0)
    mae = np.mean(np.abs(y_pred.flatten() - y_test))
    return mae

# Train Transformer Model with Bayesian Optimization
@st.cache_resource
def train_transformer_model(df, feature_cols, target_col='close'):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, df, feature_cols, target_col), n_trials=20)
    
    best_params = study.best_params
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols + [target_col], index=df.index)
    
    X, y = create_sequences(scaled_df, SEQUENCE_LENGTH, feature_cols, target_col)
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    model = build_transformer_model(
        SEQUENCE_LENGTH, len(feature_cols), best_params['num_heads'], best_params['num_layers'],
        best_params['d_model'], best_params['dropout_rate'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
    
    y_pred = model.predict(X_test, verbose=0)
    mae = np.mean(np.abs(y_pred.flatten() - y_test))
    directional_accuracy = np.mean((y_pred.flatten() > y_test) == (y_test > X_test[:, -1, feature_cols.index('close')]))
    return model, scaler, mae, directional_accuracy, best_params

# Generate Trading Signal
def generate_signal(df, signal_strategy, sentiment_score=0, x_sentiment=0):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None
    signal = "Neutral"
    pred = mae = directional_accuracy = None
    
    if signal_strategy == "EMA_RSI_MACD":
        if last['EMA_9'] > last['EMA_21'] and last['RSI'] < 70 and last['MACD'] > last['MACD_Signal']:
            signal = "Bullish"
        elif last['EMA_9'] < last['EMA_21'] and last['RSI'] > 30 and last['MACD'] < last['MACD_Signal']:
            signal = "Bearish"
    elif signal_strategy == "MA_Crossover":
        if prev is not None and last['SMA_20'] > last['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signal = "Bullish"
        elif prev is not None and last['SMA_20'] < last['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signal = "Bearish"
    elif signal_strategy == "AI_Prediction":
        feature_cols = ['close', 'volume', 'EMA_9', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
                        'pct_change', 'close_lag1', 'close_lag2', 'momentum', 'rolling_mean',
                        'rolling_std', 'volatility', 'hour', 'day_of_week', 'btc_correlation',
                        'volatility_regime', 'order_flow_imbalance', 'wavelet_trend']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        model, scaler, mae, directional_accuracy, best_params = train_transformer_model(df, feature_cols)
        scaled_data = scaler.transform(df[feature_cols].iloc[-SEQUENCE_LENGTH:])
        X_latest = np.array([scaled_data])
        pred_scaled = model.predict(X_latest, verbose=0)[0][0]
        pred = scaler.inverse_transform([[pred_scaled if col == 'close' else 0 for col in feature_cols + ['close']]])[0][-1]
        signal = "Bullish" if pred > last['close'] else "Bearish"
        
        combined_sentiment = (sentiment_score + x_sentiment) / 2
        if combined_sentiment > 0.3 and signal == "Bearish":
            signal = "Neutral"
        elif combined_sentiment < -0.3 and signal == "Bullish":
            signal = "Neutral"
    
    return signal, pred, mae, directional_accuracy

# Calculate TP and SL
def calculate_tp_sl(df, signal, tp_sl_strategy, pred=None):
    last = df.iloc[-1]
    entry = last['close']
    tp = sl = None
    
    if tp_sl_strategy == "BB":
        tp = last['BB_High'] if signal == "Bullish" else last['BB_Low']
        sl = last['BB_Low'] if signal == "Bullish" else last['BB_High']
    elif tp_sl_strategy == "ATR":
        tp = entry + 2 * last['ATR'] if signal == "Bullish" else entry - 2 * last['ATR']
        sl = entry - 1.5 * last['ATR'] if signal == "Bullish" else entry + 1.5 * last['ATR']
    elif tp_sl_strategy == "SupportResistance":
        tp = last['max'] if signal == "Bullish" else last['min']
        sl = last['min'] if signal == "Bullish" else last['max']
    elif tp_sl_strategy == "Combined":
        bb_tp = last['BB_High'] if signal == "Bullish" else last['BB_Low']
        atr_tp = entry + 2 * last['ATR'] if signal == "Bullish" else entry - 2 * last['ATR']
        sr_tp = last['max'] if signal == "Bullish" else last['min']
        tp = np.mean([bb_tp, atr_tp, sr_tp])
        bb_sl = last['BB_Low'] if signal == "Bullish" else last['BB_High']
        atr_sl = entry - 1.5 * last['ATR'] if signal == "Bullish" else entry + 1.5 * last['ATR']
        sr_sl = last['min'] if signal == "Bearish" else last['max']
        sl = np.mean([bb_sl, atr_sl, sr_sl])
    elif tp_sl_strategy == "AI" and pred is not None:
        tp = pred
        sl = entry - (tp - entry) if signal == "Bullish" else entry + (entry - tp)
    
    return round(tp, 8) if tp else None, round(sl, 8) if sl else None

# Monte Carlo Simulation for Risk Assessment
def monte_carlo_simulation(returns, n_simulations=1000):
    simulated_equity = []
    for _ in range(n_simulations):
        sim_returns = np.random.choice(returns, size=len(returns), replace=True)
        sim_equity = np.cumprod(1 + sim_returns)
        simulated_equity.append(sim_equity[-1])
    return np.percentile(simulated_equity, 5)  # 5th percentile as expected shortfall

# Backtest Signals with Walk-Forward Optimization
def backtest_signals(df, signal_strategy, tp_sl_strategy, sentiment_score=0, x_sentiment=0, lookahead=10, initial_balance=1000, fee=0.001, slippage=0.002, risk_per_trade=0.01):
    balance = initial_balance
    balance_history = [balance]
    trades = 0
    wins = 0
    rr_list = []
    returns = []
    
    # Walk-Forward Optimization
    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        for i in range(50, len(test_df) - lookahead):
            sliced = test_df.iloc[:i + 1]
            signal, pred, _, _ = generate_signal(sliced, signal_strategy, sentiment_score, x_sentiment)
            if signal not in ["Bullish", "Bearish"]:
                continue
            entry = sliced.iloc[-1]['close'] * (1 + slippage)
            atr = sliced.iloc[-1]['ATR']
            position_size = min((balance * risk_per_trade) / atr, balance / entry)
            tp, sl = calculate_tp_sl(sliced, signal, tp_sl_strategy, pred)
            if not tp or not sl:
                continue
            
            success = fail = False
            for j in range(1, lookahead + 1):
                price = test_df.iloc[i + j]['close']
                if signal == "Bullish":
                    if price >= tp:
                        success = True
                        break
                    elif price <= sl:
                        fail = True
                        break
                elif signal == "Bearish":
                    if price <= tp:
                        success = True
                        break
                    elif price >= sl:
                        fail = True
                        break
            
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk else 0
            rr_list.append(rr)
            
            if success:
                trade_return = (reward / entry) * position_size - 2 * fee
                balance += balance * trade_return
                wins += 1
            elif fail:
                trade_return = -(risk / entry) * position_size - 2 * fee
                balance += balance * trade_return
            trades += 1
            balance_history.append(balance)
            returns.append(trade_return)
    
    win_rate = round(100 * wins / trades, 2) if trades else 0
    rr_avg = round(sum(rr_list) / len(rr_list), 2) if rr_list else 0
    returns = np.array(returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) != 0 else 0
    max_drawdown = np.min(np.cumsum(returns)) * 100 if len(returns) > 0 else 0
    profit_factor = np.sum(returns[returns > 0]) / -np.sum(returns[returns < 0]) if np.sum(returns < 0) != 0 else np.inf
    expected_shortfall = monte_carlo_simulation(returns) if len(returns) > 0 else 0
    return trades, win_rate, round(balance, 2), rr_avg, balance_history, sharpe_ratio, max_drawdown, profit_factor, expected_shortfall

# Fetch News
def get_news():
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&public=true"
        news_data = requests.get(url).json()['results'][:5]
        headlines = [n['title'] for n in news_data]
    except requests.RequestException as e:
        headlines = [f"News fetch failed: {str(e)}"]
    return headlines

# Analyze Sentiment
def analyze_sentiment(news):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in news]
    return sum(scores) / len(scores) if scores else 0

# Sidebar Settings
st.sidebar.title("Settings")
selected_symbol = st.sidebar.selectbox("Select Coin", SYMBOLS)
signal_strategy = st.sidebar.selectbox("Select Signal Strategy", ["EMA_RSI_MACD", "MA_Crossover", "AI_Prediction"])
tp_sl_strategy = st.sidebar.selectbox("Select TP/SL Method", ["BB", "ATR", "SupportResistance", "Combined", "AI"])
lookahead = st.sidebar.slider("Backtest Lookahead Periods", 5, 20, 10)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0) / 100
enable_x_sentiment = st.sidebar.checkbox("Enable X Sentiment Analysis", value=False)
enable_order_book = st.sidebar.checkbox("Enable Order Book Data", value=False)

# Fetch Sentiment
news = get_news()
sentiment_score = analyze_sentiment(news)
x_sentiment = get_x_sentiment(selected_symbol) if enable_x_sentiment else 0
if enable_order_book:
    order_book_data[selected_symbol] = get_order_book(selected_symbol)

# Start WebSocket
for tf in TIMEFRAMES:
    start_websocket(selected_symbol, TIMEFRAMES[tf])

# Model Performance Dashboard
st.subheader("Model Performance Dashboard")
perf_data = []
for tf in TIMEFRAMES:
    df = realtime_data.get(selected_symbol, get_klines(selected_symbol, TIMEFRAMES[tf]))
    df = apply_indicators(df)
    for strategy in ["EMA_RSI_MACD", "MA_Crossover", "AI_Prediction"]:
        signal, pred, mae, da = generate_signal(df, strategy, sentiment_score, x_sentiment)
        t, w, final, rra, _, sharpe, mdd, pf, es = backtest_signals(
            df, strategy, tp_sl_strategy, sentiment_score, x_sentiment, lookahead, risk_per_trade=risk_per_trade)
        perf_data.append([tf, strategy, w, final, sharpe, mdd, pf, es, mae if mae else "N/A", da if da else "N/A"])
st.table(pd.DataFrame(perf_data, columns=["Timeframe", "Strategy", "Win Rate (%)", "Final Capital ($)", "Sharpe Ratio", "Max Drawdown (%)", "Profit Factor", "Expected Shortfall", "MAE", "Directional Accuracy (%)"]))

# Interactive Optimization Dashboard
st.subheader("Strategy Optimization")
with st.expander("Optimize Hyperparameters"):
    if st.button("Run Optimization"):
        df = realtime_data.get(selected_symbol, get_klines(selected_symbol, TIMEFRAMES[list(TIMEFRAMES.keys())[0]]))
        df = apply_indicators(df)
        feature_cols = ['close', 'volume', 'EMA_9', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
                        'pct_change', 'close_lag1', 'close_lag2', 'momentum', 'rolling_mean',
                        'rolling_std', 'volatility', 'hour', 'day_of_week', 'btc_correlation',
                        'volatility_regime', 'order_flow_imbalance', 'wavelet_trend']
        feature_cols = [col for col in feature_cols if col in df.columns]
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, df, feature_cols), n_trials=20)
        st.write(f"Best Hyperparameters: {study.best_params}")
        st.write(f"Best MAE: {study.best_value}")

# Main Content: Timeframe Analysis
for tf in TIMEFRAMES:
    st.subheader(f"{selected_symbol} - {tf.upper()}")
    df = realtime_data.get(selected_symbol, get_klines(selected_symbol, TIMEFRAMES[tf]))
    df = apply_indicators(df)
    signal, pred, mae, directional_accuracy = generate_signal(df, signal_strategy, sentiment_score, x_sentiment)
    entry = df.iloc[-1]['close']
    tp, sl = calculate_tp_sl(df, signal, tp_sl_strategy, pred)
    
    # Real-Time Alerts
    if signal_strategy == "AI_Prediction" and signal != "Neutral" and directional_accuracy is not None and directional_accuracy > 0.7:
        st.success(f"High-Confidence {signal} Signal on {tf.upper()}!")
    
    # Color-Coded Signal
    signal_color = {"Bullish": "green", "Bearish": "red", "Neutral": "grey"}
    st.markdown(f"**Signal:** <span style='color:{signal_color[signal]}'>{signal}</span>", unsafe_allow_html=True)
    st.write(f"**Entry:** {entry:.8f} | **TP:** {tp:.8f} | **SL:** {sl:.8f}")
    rr = round(abs(tp - entry) / abs(entry - sl), 2) if tp and sl and entry else "N/A"
    st.write(f"**Reward:Risk:** {rr}")
    if signal_strategy == "AI_Prediction" and mae is not None:
        st.write(f"**Prediction MAE:** {round(mae, 2)} | **Directional Accuracy:** {round(directional_accuracy * 100, 2)}%")
    
    # Backtesting with Enhanced Metrics
    t, w, final, rra, balance_history, sharpe, mdd, pf, es = backtest_signals(
        df, signal_strategy, tp_sl_strategy, sentiment_score, x_sentiment, lookahead, risk_per_trade=risk_per_trade)
    st.write(f"**Backtest - Trades:** {t} | **Win Rate:** {w}% | **Final Capital:** ${final} | "
             f"**Avg R:R:** {rra} | **Sharpe Ratio:** {round(sharpe, 2)} | **Max Drawdown:** {round(mdd, 2)}% | "
             f"**Profit Factor:** {round(pf, 2)} | **Expected Shortfall:** {round(es, 2)}")
    st.line_chart(balance_history)
    
    # Predicted vs Actual Price Chart
    if signal_strategy == "AI_Prediction" and pred is not None:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df.index[-20:], y=df['close'].iloc[-20:], name='Actual Price'))
        fig_pred.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + pd.Timedelta(minutes=FORECAST_HORIZON*int(tf[:-1]))],
                                      y=[entry, pred], name='Predicted Price', mode='lines+markers', line=dict(dash='dash')))
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Interactive Price Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='red')))
    st.plotly_chart(fig, use_container_width=True)

# Multi-Symbol Comparison
st.subheader("Multi-Symbol Comparison")
selected_symbols = st.multiselect("Select Coins for Comparison", SYMBOLS, default=SYMBOLS[:2])
comparison_tf = st.selectbox("Select Timeframe for Comparison", list(TIMEFRAMES.keys()))

comparison_data = []
for symbol in selected_symbols:
    df = realtime_data.get(symbol, get_klines(symbol, TIMEFRAMES[comparison_tf]))
    df = apply_indicators(df)
    x_sentiment = get_x_sentiment(symbol) if enable_x_sentiment else 0
    if enable_order_book:
        order_book_data[symbol] = get_order_book(symbol)
    signal, pred, mae, da = generate_signal(df, signal_strategy, sentiment_score, x_sentiment)
    entry = df.iloc[-1]['close']
    tp, sl = calculate_tp_sl(df, signal, tp_sl_strategy, pred)
    comparison_data.append([symbol, signal, entry, tp, sl, round(mae, 2) if mae else "N/A", round(da * 100, 2) if da else "N/A"])

st.table(pd.DataFrame(comparison_data, columns=["Symbol", "Signal", "Entry", "TP", "SL", "MAE", "Directional Accuracy (%)"]))

# Portfolio Tracker
st.subheader("Portfolio Tracker")
portfolio_symbols = st.multiselect("Select Portfolio Coins", SYMBOLS, default=SYMBOLS[:2])
initial_investment = st.number_input("Initial Investment ($)", min_value=1000.0, value=10000.0)
portfolio_data = []
for symbol in portfolio_symbols:
    df = realtime_data.get(symbol, get_klines(symbol, TIMEFRAMES['1h']))
    df = apply_indicators(df)
    t, w, final, _, _, sharpe, mdd, pf, es = backtest_signals(
        df, signal_strategy, tp_sl_strategy, sentiment_score, x_sentiment, risk_per_trade=risk_per_trade)
    portfolio_data.append([symbol, w, final, sharpe, mdd, pf, es])
st.table(pd.DataFrame(portfolio_data, columns=["Symbol", "Win Rate (%)", "Final Capital ($)", "Sharpe Ratio", "Max Drawdown (%)", "Profit Factor", "Expected Shortfall"]))

# Sentiment Analysis
st.subheader("🗞️ News and Social Sentiment")
combined_sentiment = (sentiment_score + x_sentiment) / 2 if enable_x_sentiment else sentiment_score
sentiment_color = "green" if combined_sentiment > 0 else "red" if combined_sentiment < 0 else "grey"
st.markdown(f"**Combined Sentiment Score:** <span style='color:{sentiment_color}'>{round(combined_sentiment, 2)}</span>", unsafe_allow_html=True)
st.write(f"**News Sentiment:** {round(sentiment_score, 2)} | **X Sentiment:** {round(x_sentiment, 2)}")
st.write("Recent News Headlines:")
for n in news:
    st.markdown(f"- {n}")
