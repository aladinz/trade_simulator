import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from indicators.sma import add_sma_signals

st.set_page_config(page_title="Swing Trade Simulator", layout="centered")
st.title("📈 Swing Trade Pro")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AMD)", value="AMD").upper()
capital = st.number_input("Enter Trading Capital ($)", min_value=1000, value=100000, step=1000)

# --- Run Simulation Button ---
if st.button("Run Simulation"):
    df = yf.download(ticker, period="6mo", interval="1d")

    if df.empty or df.shape[0] < 50:
        st.error("❌ Invalid or insufficient data. Try a different ticker.")
        st.stop()

    if 'Low' not in df.columns or 'High' not in df.columns:
        st.error("❌ Missing 'High' or 'Low' column in data. Cannot calculate support/resistance.")
        st.stop()

    # Add SMAs
    df = add_sma_signals(df)
    latest = df.iloc[-1]

    # Extract and clean support/resistance/stop-loss
    support = float(df['Low'].rolling(window=20).min().iloc[-1])
    resistance = float(df['High'].rolling(window=20).max().iloc[-1])
    stop_loss = round(float(latest['Close']) * 0.95, 2)

    # Extract high/low with Series-safe conversion
    low_6mo_val = df['Low'].min()
    low_6mo = float(low_6mo_val.iloc[0]) if hasattr(low_6mo_val, 'iloc') else float(low_6mo_val)

    high_6mo_val = df['High'].max()
    high_6mo = float(high_6mo_val.iloc[0]) if hasattr(high_6mo_val, 'iloc') else float(high_6mo_val)

    # Detect buy/sell signal (SMA20 crosses above SMA50)
    signal = "❌ No trade signal"
    if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2]:
        signal = "🟢 **Buy Signal Detected** (SMA20 crossed above SMA50)"
    elif df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] >= df['SMA50'].iloc[-2]:
        signal = "🔴 **Sell Signal Detected** (SMA20 crossed below SMA50)"

    # --- Display Metrics ---
    st.subheader(f"🧾 Summary for {ticker}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Open:** ${float(latest['Open']):.2f}")
        st.markdown(f"**Close:** ${float(latest['Close']):.2f}")
        st.markdown(f"**High (today):** ${float(latest['High']):.2f}")
        st.markdown(f"**Low (today):** ${float(latest['Low']):.2f}")
        st.markdown(f"**20-Day SMA:** ${float(latest['SMA20']):.2f}")

    with col2:
        st.markdown(f"**50-Day SMA:** ${float(latest['SMA50']):.2f}")
        st.markdown(f"**High (6mo):** ${high_6mo:.2f}")
        st.markdown(f"**Low (6mo):** ${low_6mo:.2f}")
        st.markdown(f"**Support (20-day):** ${support:.2f}")
        st.markdown(f"**Resistance (20-day):** ${resistance:.2f}")

    st.markdown(f"🚨 **Suggested Stop-Loss:** `${stop_loss:.2f}`")
    st.markdown(f"📉 **Signal:** {signal}")

    # --- Plot Chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=1)
    ax.plot(df.index, df['SMA20'], label='SMA20', linestyle='--', color='blue')
    ax.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='orange')
    ax.axhline(y=support, color='green', linestyle=':', linewidth=1.5, label='Support')
    ax.axhline(y=resistance, color='red', linestyle=':', linewidth=1.5, label='Resistance')

    ax.set_title(f"{ticker} Price Chart with SMA & Levels")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
