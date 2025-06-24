import os
import joblib
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

API_KEYS =  [
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    'd162b35754ca4c54a13ebe7abecab4e0',
    'a7266b2503fd497496d47527a7e63b5d',
    '09c09d58ed5e4cf4afd9a9cac8e09b5d',
    'df00920c02c54a59a426948a47095543'
]
api_index = 0
MODEL_DIR = "model"
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

def fetch_forex(symbol, interval='1h', limit=60):
    import requests
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={limit}&apikey={get_next_api_key()}"
    try:
        r = requests.get(url, timeout=10)
        d = r.json()
        if "values" not in d:
            return pd.DataFrame()
        df = pd.DataFrame(d["values"])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df['volume'] = 0
        df['spread'] = (df['high'] - df['low']) / df['close']
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def fetch_binance(symbol, interval='1h', limit=60):
    import requests
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
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
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def add_features(df):
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
    df = df.dropna()
    return df

def generate_signals(kind='forex'):
    features = [
        'close','rsi','macd','cci','adx','atr','boll_high','boll_low',
        'ma5','ma20','momentum','volatility','rolling_max','rolling
