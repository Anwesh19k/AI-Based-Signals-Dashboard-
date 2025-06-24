import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from scripts.train_module import train_model
from scripts.signal_module import generate_signals

st.set_page_config(page_title="AI Signal Dashboard", layout="wide")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

st.title("📊 AI-powered Forex & Crypto Signal Dashboard")

# Candle Counter
interval_sec = 3600  # 1h candle, change to 60 for 1min, 300 for 5min

def time_until_next_candle(interval=interval_sec):
    now = datetime.utcnow()
    epoch = int(now.timestamp())
    next_epoch = (epoch // interval + 1) * interval
    return next_epoch - epoch

st.sidebar.header("🕒 Candle Timer")
candle_time = st.sidebar.empty()
while True:
    left = time_until_next_candle(interval_sec)
    mins, secs = divmod(int(left), 60)
    candle_time.metric("Time to Next Candle", f"{mins:02d}:{secs:02d} UTC")
    time.sleep(1)
    break  # Important: Streamlit reruns, don't block forever!

# User Controls
st.header("🔧 Model Actions")

model_type = st.radio("What do you want to train/generate?", ["Forex", "Crypto", "Both"], horizontal=True)
train_btn = st.button("Train Model")
signal_btn = st.button("Generate Signals")

if train_btn:
    with st.spinner("Training in progress..."):
        result = train_model(model_type.lower())
    st.success(result)

if signal_btn:
    with st.spinner("Generating signals..."):
        df = generate_signals(model_type.lower())
        if isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Signals CSV", df.to_csv(index=False), file_name="signals.csv")
        else:
            st.warning("No signals generated yet. Please train a model first.")

st.info("Tip: Train the model first (only when you want to retrain). For new signals, just click 'Generate Signals' after every new candle.")
