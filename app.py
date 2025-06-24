import streamlit as st
from streamlit_autorefresh import st_autorefresh
import os
import pandas as pd
from datetime import datetime, timedelta

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="AI Signal Dashboard", layout="wide")

# ---- Live Candle Timer ----
st_autorefresh(interval=1000, key="candle_timer_refresh")
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

# ---- Model Directory Tools ----
st.sidebar.subheader("🗂 Model Directory Tools")

def list_model_files():
    files = os.listdir(MODEL_DIR)
    if not files:
        st.sidebar.info("No files in model directory.")
    else:
        st.sidebar.write("**Files in /model:**")
        for fname in files:
            st.sidebar.code(fname)

def clean_model_dir():
    deleted = []
    for fname in os.listdir(MODEL_DIR):
        if fname == ".gitkeep":
            continue
        fpath = os.path.join(MODEL_DIR, fname)
        try:
            os.remove(fpath)
            deleted.append(fname)
        except Exception as e:
            st.sidebar.warning(f"Could not delete {fname}: {e}")
    return deleted

list_model_files()

if st.sidebar.button("🧹 Clean model directory"):
    deleted = clean_model_dir()
    if deleted:
        st.sidebar.success(f"Deleted: {', '.join(deleted)}")
    else:
        st.sidebar.info("Nothing to delete.")
    list_model_files()

# ---- Main Dashboard ----
st.title("📊 AI-powered Forex & Crypto Signal Dashboard")
st.header("🔧 Model Actions")

model_type = st.radio("What do you want to train/generate?", ["Forex", "Crypto", "Both"], horizontal=True)
train_btn = st.button("Train Model")
signal_btn = st.button("Generate Signals")

def train_model(kind='forex'):
    # Placeholder logic (replace with your actual train_module.py logic)
    return f"Training {kind} models... (placeholder)"

def generate_signals(kind='forex'):
    # Placeholder DataFrame (replace with your signal_module.py logic)
    data = {
        "SYMBOL": ["EUR/USD", "USD/JPY"],
        "SIGNAL": ["BUY 📈", "SELL 🔻"],
        "CONFIDENCE": ["0.73", "0.51"],
        "MODEL ACCURACY": ["62.4%", "57.0%"],
        "LAST PRICE": ["1.1701", "145.3600"]
    }
    return pd.DataFrame(data)

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

st.info("Tip: Train the model first. After each new candle, click 'Generate Signals' to update.")

