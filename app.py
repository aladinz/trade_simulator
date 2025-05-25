import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from datetime import datetime

from indicators.sma import add_sma_signals

st.set_page_config(page_title="Swing Trade Simulator", layout="centered")
st.title("📈 Swing Trade Pro")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AMD)", value="AMD").upper()
capital = st.number_input("Enter Trading Capital ($)", min_value=1000, value=100000, step=1000)

# --- Run Simulation Button ---
if st.button("Run Simulation"):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.reset_index(inplace=True)

    if df.empty or df.shape[0] < 50:
        st.error("❌ Invalid or insufficient data. Try a different ticker.")
        st.stop()

    if 'Low' not in df.columns or 'High' not in df.columns:
        st.error("❌ Missing 'High' or 'Low' column in data. Cannot calculate support/resistance.")
        st.stop()

    df = add_sma_signals(df)
    latest = df.iloc[-1]

    support = float(df['Low'].rolling(window=20).min().iloc[-1])
    resistance = float(df['High'].rolling(window=20).max().iloc[-1])
    stop_loss = round(float(latest['Close']) * 0.95, 2)

    low_6mo = float(df['Low'].min())
    high_6mo = float(df['High'].max())

    signals = []
    for i in range(1, len(df)):
        price_now = float(df['Close'].iloc[i])
        price_prev = float(df['Close'].iloc[i - 1])
        sma20_now = float(df['SMA20'].iloc[i])
        sma20_prev = float(df['SMA20'].iloc[i - 1])
        sma50_now = float(df['SMA50'].iloc[i])
        sma50_prev = float(df['SMA50'].iloc[i - 1])

        if not any(pd.isna([price_now, price_prev, sma20_now, sma20_prev, sma50_now, sma50_prev])):
            if sma20_prev < sma50_prev and sma20_now > sma50_now and price_now > sma20_now > sma50_now:
                signals.append((df['Date'].iloc[i], "🟢 Buy"))
            elif sma20_prev > sma50_prev and sma20_now < sma50_now:
                signals.append((df['Date'].iloc[i], "🔴 Sell"))

    signal_text = "❌ No trade signal"
    if signals:
        last_signal = signals[-1]
        signal_text = f"{last_signal[1]} Signal Detected on {last_signal[0].date()}"

    trend_text = "📊 **Market Trend:** 🟢 Bullish (SMA20 above SMA50)" if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] else "📊 **Market Trend:** 🔴 Not Bullish (SMA20 below SMA50)"

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
    st.markdown(f"📉 **Signal:** {signal_text}")
    st.markdown(trend_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label='Close', color='black', linewidth=1)
    ax.plot(df['Date'], df['SMA20'], label='SMA20', linestyle='--', color='blue')
    ax.plot(df['Date'], df['SMA50'], label='SMA50', linestyle='--', color='orange')
    ax.axhline(y=support, color='green', linestyle=':', linewidth=1.5, label='Support')
    ax.axhline(y=resistance, color='red', linestyle=':', linewidth=1.5, label='Resistance')

    for signal_date, signal_type in signals:
        y = df.loc[df['Date'] == signal_date, 'Close'].values[0]
        color = 'green' if "Buy" in signal_type else 'red'
        marker = '^' if "Buy" in signal_type else 'v'
        ax.plot(signal_date, y, marker=marker, color=color, markersize=8)
        ax.text(signal_date, y, signal_type, color=color, fontsize=8, ha='center')

    recent_df = df.tail(30).copy()
    recent_df['Date_Ordinal'] = recent_df['Date'].map(pd.Timestamp.toordinal)
    X = recent_df[['Date_Ordinal']].values
    y = recent_df['Close'].values
    poly_X = np.hstack([X, X**2])
    model = LinearRegression()
    model.fit(poly_X, y)

    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=65, freq='B')
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_poly = np.hstack([future_ordinals, future_ordinals**2])
    future_preds = model.predict(future_poly)
    ax.plot(future_dates, future_preds, label='3M Forecast', color='purple', linestyle=':')

    ax.set_title(f"{ticker} Price Chart with SMA & Levels")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Analyst Consensus Rating (real-time from Yahoo Finance - fallback to info) ---
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if "recommendationKey" in info:
            st.markdown("### 🧠 Analyst Consensus (from Yahoo Finance)")
            st.markdown(f"- **Recommendation**: {info['recommendationKey'].capitalize()}")
            if "numberOfAnalystOpinions" in info:
                st.markdown(f"- **Total Analysts**: {info['numberOfAnalystOpinions']}")
            if all(k in info for k in ["targetLowPrice", "targetMeanPrice", "targetHighPrice"]):
                st.markdown("**Price Target Range:**")
                st.markdown(f"- **Low Target**: ${info['targetLowPrice']:.2f}")
                st.markdown(f"- **Average Target**: ${info['targetMeanPrice']:.2f}")
                st.markdown(f"- **High Target**: ${info['targetHighPrice']:.2f}")
        else:
            st.info("ℹ️ No analyst recommendation data available for this ticker.")
    except Exception as e:
        st.error(f"Failed to load analyst data: {e}")
     

# --- Recent News Highlights (live using Finnhub API) ---
    st.markdown("### 🗞️ Recent News Highlights")
    try:
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2024-05-01&to=2025-05-23&token=d0p5nv1r01qr8ds1mcn0d0p5nv1r01qr8ds1mcng"
        res = requests.get(news_url)
        if res.status_code == 200:
            articles = res.json()
            if articles:
                for article in articles[:3]:
                    headline = article.get("headline")
                    datetime_ts = datetime.fromtimestamp(article.get("datetime"))
                    date_str = datetime_ts.strftime("%Y-%m-%d %H:%M")
                    url = article.get("url")

                    # Simple sentiment tagging
                    headline_lower = headline.lower()
                    if any(x in headline_lower for x in ["beats", "soars", "jumps", "record", "growth"]):
                        sentiment = "🟢 Bullish"
                    elif any(x in headline_lower for x in ["misses", "drops", "falls", "loss", "concern"]):
                        sentiment = "🔴 Bearish"
                    else:
                        sentiment = "⚪ Neutral"

                    # Highlight earnings-related headlines
                    if "earnings" in headline_lower or "q" in headline_lower and "report" in headline_lower:
                        headline_display = f"**📣 {headline}**"
                    else:
                        headline_display = headline

                    st.markdown(f"- [{headline_display}]({url})  {sentiment} ⏱️ {date_str}")
            else:
                st.info("ℹ️ No recent news articles available for this ticker.")
        else:
            st.warning("⚠️ Could not fetch news from Finnhub. Status code: " + str(res.status_code))
    except Exception as e:
        st.error(f"Failed to fetch news headlines: {e}")