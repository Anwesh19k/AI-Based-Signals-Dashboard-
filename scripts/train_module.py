import os
import joblib
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

API_KEYS = [
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    'd162b35754ca4c54a13ebe7abecab4e0',
    'a7266b2503fd497496d47527a7e63b5d',
    '09c09d58ed5e4cf4afd9a9cac8e09b5d',
    'df00920c02c54a59a426948a47095543'
]
api_index = 0
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

FOREX_SYMBOLS = [
    'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD',
    'USD/CAD', 'NZD/USD', 'EUR/GBP', 'AUD/JPY', 'GBP/JPY'
]
CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT']

def get_next_api_key():
    global api_index
    key = API_KEYS[api_index % len(API_KEYS)]
    api_index += 1
    return key

def fetch_forex(symbol, interval='1h', limit=2000):
    import requests
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={limit}&apikey={get_next_api_key()}"
    try:
        r = requests.get(url, timeout=15)
        d = r.json()
        if "values" not in d:
            print(f"{symbol}: API error: {d.get('message', r.text)}")
            return pd.DataFrame()
        df = pd.DataFrame(d["values"])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df['volume'] = 0
        df['spread'] = (df['high'] - df['low']) / df['close']
        return df
    except Exception as e:
        print(f"{symbol}: Exception: {e}")
        return pd.DataFrame()

def fetch_binance(symbol, interval='1h', limit=2000):
    import requests
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            print(f"{symbol}: API error (Binance): {r.text}")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'open_time','open','high','low','close','volume','close_time','qav',
            'num_trades','taker_base_vol','taker_quote_vol','ignore'
        ])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df['spread'] = (df['high'] - df['low']) / df['close']
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'spread']]
    except Exception as e:
        print(f"{symbol}: Exception: {e}")
        return pd.DataFrame()

def add_features(df, hold_ahead=4, return_threshold=0.0007):
    df = df.copy()
    if len(df) < 50: return df
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    bb = BollingerBands(df['close'])
    df['boll_high'] = bb.bollinger_hband()
    df['boll_low'] = bb.bollinger_lband()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['volatility'] = df['close'].rolling(20).std()
    df['rolling_max'] = df['close'].rolling(10).max()
    df['rolling_min'] = df['close'].rolling(10).min()
    df['zscore'] = (df['close'] - df['ma20']) / (df['volatility']+1e-8)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_5'] = df['log_ret'].rolling(5).std()
    df['vol_10'] = df['log_ret'].rolling(10).std()
    df['pos_in_range'] = (df['close'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'] + 1e-8)
    df['hour'] = df['datetime'].dt.hour
    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] <= 15)).astype(int)
    df['is_ny'] = ((df['hour'] >= 12) & (df['hour'] <= 20)).astype(int)
    df['future_close'] = df['close'].shift(-hold_ahead)
    df['future_ret'] = (df['future_close'] - df['close']) / df['close']
    df['label'] = (df['future_ret'] > return_threshold).astype(int)
    df = df.dropna()
    return df

def walk_forward_validation(X, y, features, model, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_val_scaled, y_val)
        scores.append(score)
    return np.mean(scores)

def train_model(kind='forex'):
    features = [
        'close','rsi','macd','cci','adx','atr','boll_high','boll_low',
        'ma5','ma20','momentum','volatility','rolling_max','rolling_min','zscore','spread',
        'log_ret','vol_5','vol_10','pos_in_range','is_london','is_ny'
    ]
    status_list = []
    if kind in ['forex', 'both']:
        for symbol in FOREX_SYMBOLS:
            print(f"Training {symbol} ...")
            df = fetch_forex(symbol)
            if df.empty or len(df) < 200:
                status_list.append(f"{symbol}: Not enough data or API issue.")
                continue
            df = add_features(df)
            if len(df['label'].unique()) < 2:
                status_list.append(f"{symbol}: Only one class present after feature engineering.")
                continue
            X = df[features].values
            y = df['label'].values
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)
            rf = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42)
            rf_cv = walk_forward_validation(X, y, features, rf)
            rf.fit(X_scaled, y)
            symbol_key = symbol.replace('/','')
            joblib.dump(rf, os.path.join(MODEL_DIR, f"{symbol_key}_rf.joblib"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, f"{symbol_key}_scaler.joblib"))
            with open(os.path.join(MODEL_DIR, f"{symbol_key}_rf_cv.txt"), "w") as f:
                f.write(str(rf_cv))
            status_list.append(f"{symbol}: Trained, CV Acc: {rf_cv:.2%}")
    if kind in ['crypto', 'both']:
        for symbol in CRYPTO_SYMBOLS:
            print(f"Training {symbol} ...")
            df = fetch_binance(symbol)
            if df.empty or len(df) < 200:
                status_list.append(f"{symbol}: Not enough data or API issue.")
                continue
            df = add_features(df)
            if len(df['label'].unique()) < 2:
                status_list.append(f"{symbol}: Only one class present after feature engineering.")
                continue
            X = df[features].values
            y = df['label'].values
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)
            rf = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42)
            rf_cv = walk_forward_validation(X, y, features, rf)
            rf.fit(X_scaled, y)
            joblib.dump(rf, os.path.join(MODEL_DIR, f"{symbol}_rf.joblib"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib"))
            with open(os.path.join(MODEL_DIR, f"{symbol}_rf_cv.txt"), "w") as f:
                f.write(str(rf_cv))
            status_list.append(f"{symbol}: Trained, CV Acc: {rf_cv:.2%}")
    return "\n".join(status_list)
