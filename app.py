import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Auto-refresh for candle timer ---
st_autorefresh(interval=1000, key="candle_timer_refresh")

st.set_page_config(page_title="AI Signal Dashboard", layout="wide")

st.title("📊 AI-powered Forex & Crypto Signal Dashboard")

# --- Live 1-hour Candle Timer ---
st.sidebar.header("🕒 1-Hour Candle Timer")
def seconds_to_next_hour():
    now = datetime.utcnow()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    secs_left = int((next_hour - now).total_seconds())
    return secs_left

secs_left = seconds_to_next_hour()
mins, secs = divmod(secs_left, 60)
st.sidebar.markdown(
    f"""
    <div style="font-size:2.5em; font-weight:bold; text-align:center; color:#16d">
        {mins:02d}:{secs:02d} (UTC)
    </div>
    <div style="text-align:center;">until next H1 candle</div>
    """,
    unsafe_allow_html=True
)

# --- User Controls ---
st.header("🔧 Model Actions")
model_type = st.radio("What do you want to train/generate?", ["Forex", "Crypto", "Both"], horizontal=True)
train_btn = st.button("Train Model")
signal_btn = st.button("Generate Signals")

# --- Import modular code ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.train_module import train_model
from scripts.signal_module import generate_signals

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

if train_btn:
    with st.spinner("Training in progress..."):
        result = train_model(model_type.lower())
    st.success(result)

if signal_btn:
    with st.spinner("Generating signals..."):
        df = generate_signals(model_type.lower())
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Signals CSV", df.to_csv(index=False), file_name="signals.csv")
        else:
            st.warning("No signals generated yet. Please train a model first.")

st.info("Tip: Train the model first. After each new candle, click 'Generate Signals' to update.")

