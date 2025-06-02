import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import requests
from datetime import datetime
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators.sma import add_sma_signals
# Import from the fixed predictor module
from predictor_fixed import predict_3_month_forecast, get_feature_importance
# Import trading strategies module
from trading_strategies import (
    TradingStrategy, StrategyBacktester, PortfolioManager, RiskManager, 
    StrategyTemplates, quick_backtest, calculate_optimal_position_size
)

# Set page configuration with a modern theme
st.set_page_config(
    page_title="Swing Trade Pro | Advanced Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .card {
        border-radius: 0.5rem;
        background-color: #ffffff;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }    .positive {color: #4CAF50;}
    .negative {color: #F44336;}
    .neutral {color: #9E9E9E;}
    .strategy-info {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .backtest-metrics {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        min-width: 120px;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 0.3rem;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App Header with Logo
st.markdown('<div class="main-header">📈 Swing Trade Pro</div>', unsafe_allow_html=True)

# Create a clean, modern sidebar
with st.sidebar:
    st.markdown("### 🛠️ Settings")
    
    # Put settings in a nice card-like container
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ticker = st.text_input("Stock Ticker Symbol", value="AMD").upper()
        capital = st.number_input("Trading Capital ($)", min_value=1000, value=100000, step=1000)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a Run button with improved styling
    if st.button("🚀 Run Analysis", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

# --- Run Simulation When Button is Pressed ---
if st.session_state.run_analysis:
    # Show loading spinner while fetching data
    with st.spinner("Fetching market data and analyzing patterns..."):
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
        
        # Get company information for display
        try:
            company = yf.Ticker(ticker)
            company_info = company.info
            company_name = company_info.get('longName', ticker)
            sector = company_info.get('sector', 'N/A')
            industry = company_info.get('industry', 'N/A')
        except:
            company_name = ticker
            sector = "N/A"
            industry = "N/A"

    # Calculate key metrics
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
        signal_text = f"{last_signal[1]} Signal Detected on {last_signal[0].date()}"    # Use scalar comparison to avoid Series ambiguity
    sma20_last = float(df['SMA20'].iloc[-1])
    sma50_last = float(df['SMA50'].iloc[-1])
    trend_text = "🟢 Bullish (SMA20 above SMA50)" if sma20_last > sma50_last else "🔴 Bearish (SMA20 below SMA50)"
      # Modern layout with tabs
    tabs = st.tabs(["📊 Market Analysis", "🧠 ML Insights", "📰 News & Sentiment", "📈 Strategy Backtesting"])
    
    with tabs[0]:
        # Company header
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>{company_name} ({ticker})</h2>", unsafe_allow_html=True)
            if sector != "N/A" and industry != "N/A":
                st.markdown(f"<p style='text-align: center;'><b>Sector:</b> {sector} | <b>Industry:</b> {industry}</p>", unsafe_allow_html=True)
        
        # Create cards for key metrics in 4 columns
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns(4)
        
        # Calculate price change
        price_change = float(latest['Close']) - float(latest['Open'])
        pct_change = (price_change / float(latest['Open'])) * 100
        
        # Display key metrics
        with cols[0]:
            st.markdown("<p class='metric-label'>Current Price</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>${float(latest['Close']):.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='{'positive' if price_change >= 0 else 'negative'}'>{'+' if price_change >= 0 else ''}{price_change:.2f} ({pct_change:.2f}%)</p>", unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("<p class='metric-label'>Today's Range</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>${float(latest['Low']):.2f} - ${float(latest['High']):.2f}</p>", unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown("<p class='metric-label'>Volume</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{int(latest.get('Volume', 0)):,}</p>", unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown("<p class='metric-label'>Market Trend</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value {'positive' if 'Bullish' in trend_text else 'negative'}'>{trend_text}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical indicators section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>📉 Technical Indicators</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p class='metric-label'>Moving Averages</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>SMA20:</b> <span class='metric-value'>${float(latest['SMA20']):.2f}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>SMA50:</b> <span class='metric-value'>${float(latest['SMA50']):.2f}</span></p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<p class='metric-label'>Support & Resistance</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Support:</b> <span class='metric-value positive'>${support:.2f}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Resistance:</b> <span class='metric-value negative'>${resistance:.2f}</span></p>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<p class='metric-label'>Trade Signals</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Signal:</b> <span class='metric-value'>{signal_text}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Stop-Loss:</b> <span class='metric-value negative'>${stop_loss:.2f}</span></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Create the chart with improved styling        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>📈 Price Chart with Technical Analysis</h3>", unsafe_allow_html=True)
        
        # Create a well-styled matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('ggplot')
        
        # Configure date formatting for the x-axis
        from matplotlib.dates import DateFormatter, MonthLocator
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator())
        
        # Plot price and indicators
        ax.plot(df['Date'], df['Close'], label='Close', color='#2C3E50', linewidth=1.5)
        ax.plot(df['Date'], df['SMA20'], label='SMA20', linestyle='--', color='#3498DB', linewidth=1.2)
        ax.plot(df['Date'], df['SMA50'], label='SMA50', linestyle='--', color='#F39C12', linewidth=1.2)
        ax.axhline(y=support, color='#2ECC71', linestyle=':', linewidth=1.2, label='Support')
        ax.axhline(y=resistance, color='#E74C3C', linestyle=':', linewidth=1.2, label='Resistance')        # Add buy/sell signals
        for signal_date, signal_type in signals:
            y = df.loc[df['Date'] == signal_date, 'Close'].values[0]
            color = '#2ECC71' if "Buy" in signal_type else '#E74C3C'
            marker = '^' if "Buy" in signal_type else 'v'
            ax.plot(signal_date, y, marker=marker, color=color, markersize=10)
            ax.text(signal_date, y * 1.01, signal_type.split()[0], color=color, fontsize=9, ha='center')
        
        # Add ML-based forecast line
        try:
            # Get forecast from predictor module
            forecast_result = predict_3_month_forecast(df, model_type='ensemble')
            forecast_dates = forecast_result['dates']
            forecast_values = forecast_result['forecast']
            confidence_intervals = forecast_result['confidence_intervals']
            
            # Make sure forecast dates are in the correct format for plotting
            if not isinstance(forecast_dates[0], pd.Timestamp):
                forecast_dates = pd.to_datetime(forecast_dates)
                
            # For debugging or informational purposes
            with st.expander("ML Forecast Details"):
                st.write(f"Forecast length: {len(forecast_dates)} days")
                st.write(f"Date range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
                st.write(f"Value range: ${min(forecast_values):.2f} to ${max(forecast_values):.2f}")
              # Plot the forecast line with highest visibility to ensure it's on top
            ml_line = ax.plot(forecast_dates, forecast_values, label='3M ML Forecast', 
                             color='#7D3C98', linestyle='-.', linewidth=3.5, zorder=100)
              # Plot confidence intervals as shaded area with improved visibility
            lower_bounds = [ci[0] for ci in confidence_intervals]
            upper_bounds = [ci[1] for ci in confidence_intervals]
            conf_area = ax.fill_between(forecast_dates, lower_bounds, upper_bounds, 
                                     color='#9B59B6', alpha=0.4, zorder=90,
                                     label='Confidence Interval')
                                     
            # Add a boundary line around the confidence interval for better visibility
            ax.plot(forecast_dates, lower_bounds, color='#7D3C98', linestyle=':', linewidth=1.0, zorder=95)
            ax.plot(forecast_dates, upper_bounds, color='#7D3C98', linestyle=':', linewidth=1.0, zorder=95)
              # Add a legend entry manually to ensure it appears
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='#7D3C98', lw=3.5, linestyle='-.'),
                Line2D([0], [0], marker='s', color='#9B59B6', lw=0, markerfacecolor='#9B59B6', 
                       alpha=0.3, markersize=10)
            ]
            custom_labels = ['3M ML Forecast', 'Confidence Interval']
            
            # Make sure we have an explicit legend entry for the ML forecast
            ax.plot([], [], color='#7D3C98', linestyle='-.', linewidth=3.5, 
                    label='3M ML Forecast')
            
            # Store these for use in the legend later
            if not hasattr(ax, 'custom_handles'):
                ax.custom_handles = []
                ax.custom_labels = []
            ax.custom_handles.extend(custom_lines)
            ax.custom_labels.extend(custom_labels)
            
            # Add forecast metrics as an annotation on the chart
            metrics = forecast_result['model_metrics']
            ax.annotate(f"ML Forecast RMSE: ${metrics['rmse']:.2f}", 
                       xy=(0.02, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                       fontsize=9)
            
            # Ensure the forecast is visible by extending the x-axis if needed
            ax.set_xlim(right=forecast_dates[-1])
        except Exception as e:
            st.warning(f"Could not generate ML forecast: {str(e)}")
            import traceback
            st.write(f"Error details: {traceback.format_exc()}")
            # Fallback to simple polynomial regression if ML forecast fails
            recent_df = df.tail(30).copy()
            recent_df['Date_Ordinal'] = recent_df['Date'].map(pd.Timestamp.toordinal)
            X = recent_df[['Date_Ordinal']].values
            y = recent_df['Close'].values
            poly_X = np.hstack([X, X**2])
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(poly_X, y)

            future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=65, freq='B')
            future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            future_poly = np.hstack([future_ordinals, future_ordinals**2])
            future_preds = model.predict(future_poly)
            ax.plot(future_dates, future_preds, label='3M Simple Forecast', color='#9B59B6', linestyle='-.', linewidth=1.5)

        # Improve chart styling        ax.set_title(f"{ticker} Price Analysis with Moving Averages", fontsize=14, pad=20)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
          # Create comprehensive legend with ML forecast always visible
        handles, labels = ax.get_legend_handles_labels()
            
        # Create the legend with all handles, using a larger and more visible font
        if handles:
            # Make sure all legend entries are unique
            by_label = dict(zip(labels, handles))
            unique_labels = list(by_label.keys())
            unique_handles = list(by_label.values())
            
            # Explicitly ensure ML forecast is in a prominent position
            if '3M ML Forecast' in unique_labels:
                ml_index = unique_labels.index('3M ML Forecast')
                # Move ML forecast to the top of the legend
                unique_labels.insert(0, unique_labels.pop(ml_index))
                unique_handles.insert(0, unique_handles.pop(ml_index))
                
            # Create an enhanced legend
            legend = ax.legend(unique_handles, unique_labels, frameon=True, loc='upper left',
                        fontsize=11, fancybox=True, shadow=True, framealpha=0.9,
                        title='Legend', title_fontsize=12)
        
        # Add some padding to the chart
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyst Consensus Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🧠 Analyst Consensus</h3>", unsafe_allow_html=True)
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            if "recommendationKey" in info:
                rec_key = info['recommendationKey'].capitalize()
                rec_color = {
                    'Buy': 'positive',
                    'Strong_buy': 'positive',
                    'Hold': 'neutral',
                    'Sell': 'negative',
                    'Strong_sell': 'negative'
                }.get(info['recommendationKey'], 'neutral')
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"<p class='metric-label'>Recommendation</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='metric-value {rec_color}'>{rec_key.replace('_', ' ')}</p>", unsafe_allow_html=True)
                    
                    if "numberOfAnalystOpinions" in info:
                        st.markdown(f"<p>Based on {info['numberOfAnalystOpinions']} analyst opinions</p>", unsafe_allow_html=True)
                
                if all(k in info for k in ["targetLowPrice", "targetMeanPrice", "targetHighPrice"]):
                    with col2:
                        st.markdown("<p class='metric-label'>Price Target Range</p>", unsafe_allow_html=True)
                        
                        # Create a simple chart for price targets
                        current = float(latest['Close'])
                        low = info['targetLowPrice']
                        mean = info['targetMeanPrice']
                        high = info['targetHighPrice']
                        
                        st.markdown(f"""
                        <div style='display: flex; align-items: center;'>
                            <div style='width: 30%;'><b>Low:</b> <span class='metric-value'>${low:.2f}</span></div>
                            <div style='width: 40%;'><b>Average:</b> <span class='metric-value'>${mean:.2f}</span></div>
                            <div style='width: 30%;'><b>High:</b> <span class='metric-value'>${high:.2f}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show current price vs targets
                        pct_to_target = (mean - current) / current * 100
                        direction = "upside" if pct_to_target > 0 else "downside"
                        st.markdown(f"<p>Current price has <span class='{'positive' if pct_to_target > 0 else 'negative'}'>{abs(pct_to_target):.1f}% {direction}</span> to average target</p>", unsafe_allow_html=True)
            else:
                st.info("ℹ️ No analyst recommendation data available.")
        except Exception as e:
            st.error(f"Failed to load analyst data: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)    # ML Insights tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🧠 Machine Learning Predictions</h3>", unsafe_allow_html=True)

        # Get feature importance
        try:
            feature_importance = get_feature_importance(df, model_type='ensemble')
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("<p class='metric-label'>Feature Importance</p>", unsafe_allow_html=True)
                
                # Create bar chart for feature importance
                fig, ax = plt.subplots(figsize=(10, 8))
                features = [f[0] for f in sorted_features[:10]]  # Top 10 features
                importance = [f[1] for f in sorted_features[:10]]
                
                bars = ax.barh(features, importance, color='#3498DB')
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Features for Price Prediction')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add value labels to bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("<p class='metric-label'>Prediction Metrics</p>", unsafe_allow_html=True)
                
                try:
                    # Train model and get metrics
                    forecast_result = predict_3_month_forecast(df, model_type='ensemble')
                    metrics = forecast_result['model_metrics']
                    
                    # Display metrics in a nice format
                    st.markdown(f"""
                    <div style='padding: 10px; background-color: rgba(52, 152, 219, 0.1); border-radius: 5px;'>
                        <p><b>RMSE:</b> ${metrics['rmse']:.2f}</p>
                        <p><b>MAE:</b> ${metrics['mae']:.2f}</p>
                        <p><b>Confidence Level:</b> 95%</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating ML predictions: {str(e)}")
                    st.markdown("""
                    <div style='padding: 10px; background-color: rgba(231, 76, 60, 0.1); border-radius: 5px;'>
                        <p>Could not generate prediction metrics.</p>
                        <p>Please try a different ticker or timeframe.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<p class='metric-label'>Model Information</p>", unsafe_allow_html=True)
                st.markdown("""
                <div style='padding: 10px; background-color: rgba(52, 152, 219, 0.1); border-radius: 5px;'>
                    <p><b>Model Type:</b> Ensemble (Random Forest + XGBoost)</p>
                    <p><b>Features:</b> 16 technical and time-based indicators</p>
                    <p><b>Forecast Horizon:</b> 90 days</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not generate ML insights: {str(e)}")
            st.info("Try a different ticker or timeframe with more historical data.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # News tab
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🗞️ Recent News Highlights</h3>", unsafe_allow_html=True)
        
        try:
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2024-05-01&to=2025-05-23&token=d0p5nv1r01qr8ds1mcn0d0p5nv1r01qr8ds1mcng"
            res = requests.get(news_url)
            if res.status_code == 200:
                articles = res.json()
                if articles:
                    for article in articles[:6]:
                        headline = article.get("headline")
                        datetime_ts = datetime.fromtimestamp(article.get("datetime"))
                        date_str = datetime_ts.strftime("%Y-%m-%d %H:%M")
                        url = article.get("url")

                        # Simple sentiment tagging
                        headline_lower = headline.lower()
                        if any(x in headline_lower for x in ["beats", "soars", "jumps", "record", "growth"]):
                            sentiment = "🟢 Bullish"
                            sentiment_class = "positive"
                        elif any(x in headline_lower for x in ["misses", "drops", "falls", "loss", "concern"]):
                            sentiment = "🔴 Bearish"
                            sentiment_class = "negative"
                        else:
                            sentiment = "⚪ Neutral"
                            sentiment_class = "neutral"

                        # Highlight earnings-related headlines
                        if "earnings" in headline_lower or ("q" in headline_lower and "report" in headline_lower):
                            headline = f"📣 {headline}"
                            
                        # Create a styled news card
                        st.markdown(f"""
                        <div style='margin-bottom: 15px; padding: 10px; border-left: 4px solid #1E88E5; background-color: rgba(30, 136, 229, 0.05);'>
                            <a href="{url}" target="_blank" style='color: #1E88E5; font-weight: bold; text-decoration: none;'>{headline}</a>
                            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                                <span class='{sentiment_class}' style='font-weight: 600;'>{sentiment}</span>
                                <span style='color: #666; font-size: 0.9rem;'>⏱️ {date_str}</span>
                            </div>
                        </div>                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No recent news articles available for this ticker.")
            else:
                st.warning(f"⚠️ Could not fetch news from Finnhub. Status code: {res.status_code}")
        except Exception as e:
            st.error(f"Failed to fetch news headlines: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Strategy Backtesting tab
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>📈 Trading Strategy Backtesting</h3>", unsafe_allow_html=True)
        
        # Strategy selection section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 Strategy Selection")
            
            strategy_options = {
                "RSI Mean Reversion": "rsi_mean_reversion",
                "SMA Crossover": "sma_crossover", 
                "Bollinger Band Breakout": "bollinger_band_breakout",
                "Momentum Combo": "momentum_combo"
            }
            
            selected_strategy_name = st.selectbox(
                "Choose a trading strategy:",
                options=list(strategy_options.keys()),
                help="Select a predefined trading strategy to backtest"
            )
            
            selected_strategy = strategy_options[selected_strategy_name]
            
            # Capital allocation
            backtest_capital = st.number_input(
                "Backtest Capital ($)", 
                min_value=10000, 
                value=capital, 
                step=5000,
                help="Initial capital for backtesting"
            )
            
            # Risk settings
            st.markdown("### ⚠️ Risk Management")
            custom_risk = st.checkbox("Customize Risk Settings", value=False)
            
            if custom_risk:
                stop_loss_pct = st.slider("Stop Loss (%)", 1, 15, 5, help="Maximum loss per trade")
                take_profit_pct = st.slider("Take Profit (%)", 5, 30, 12, help="Target profit per trade")
                risk_per_trade = st.slider("Risk per Trade (%)", 1, 5, 2, help="Portfolio risk per trade")
            else:
                stop_loss_pct = 5
                take_profit_pct = 12
                risk_per_trade = 2
        
        with col2:
            st.markdown("### 📊 Strategy Information")
            
            # Display strategy details
            if selected_strategy == "rsi_mean_reversion":
                st.info("""
                **RSI Mean Reversion Strategy**
                - Entry: RSI < 30 (oversold)
                - Exit: RSI > 70 (overbought)
                - Best for: Range-bound markets
                - Risk: Moderate
                """)
            elif selected_strategy == "sma_crossover":
                st.info("""
                **SMA Crossover Strategy**
                - Entry: SMA20 crosses above SMA50
                - Exit: SMA20 crosses below SMA50
                - Best for: Trending markets
                - Risk: Low-Medium
                """)
            elif selected_strategy == "bollinger_band_breakout":
                st.info("""
                **Bollinger Band Breakout**
                - Entry: Price near lower band
                - Exit: Price near upper band
                - Best for: Volatile markets
                - Risk: Medium
                """)
            else:  # momentum_combo
                st.info("""
                **Momentum Combo Strategy**
                - Entry: RSI>50, SMA bullish, MACD>0
                - Exit: RSI<40 or SMA bearish
                - Best for: Strong trends
                - Risk: Medium-High
                """)
            
            # Position sizing calculator
            st.markdown("### 🧮 Position Sizing")
            current_price = float(latest['Close'])
            position_calc = calculate_optimal_position_size(
                portfolio_value=backtest_capital,
                current_price=current_price,
                stop_loss_pct=stop_loss_pct/100,
                risk_pct=risk_per_trade/100
            )
            
            st.markdown(f"""
            **Recommended Position:**
            - Shares: {position_calc['recommended_shares']:,}
            - Position Value: ${position_calc['position_value']:,.2f}
            - Portfolio Weight: {position_calc['position_weight_pct']:.1f}%
            - Max Risk: ${position_calc['max_potential_loss']:.2f}
            """)
        
        # Run backtest button
        if st.button("🚀 Run Backtest", key="run_backtest", use_container_width=True):
            with st.spinner("Running strategy backtest..."):
                try:
                    # Run the backtest
                    backtest_results = quick_backtest(
                        df=df, 
                        strategy_name=selected_strategy,
                        initial_capital=backtest_capital
                    )
                    
                    # Display results
                    st.markdown("### 📈 Backtest Results")
                    
                    # Key metrics in columns
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        total_return = backtest_results['total_return_pct']
                        color = "positive" if total_return > 0 else "negative"
                        st.markdown(f"""
                        <div class="card">
                            <p class="metric-label">Total Return</p>
                            <p class="metric-value {color}">{total_return:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        win_rate = backtest_results['win_rate_pct']
                        color = "positive" if win_rate > 50 else "negative"
                        st.markdown(f"""
                        <div class="card">
                            <p class="metric-label">Win Rate</p>
                            <p class="metric-value {color}">{win_rate:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        num_trades = backtest_results['num_trades']
                        # Always use neutral color for total trades count to ensure visibility
                        st.markdown(f"""
                        <div class="card">
                            <p class="metric-label">Total Trades</p>
                            <p class="metric-value neutral">{num_trades}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
                        color = "positive" if sharpe_ratio > 1 else "negative"
                        st.markdown(f"""
                        <div class="card">
                            <p class="metric-label">Sharpe Ratio</p>
                            <p class="metric-value {color}">{sharpe_ratio:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Performance chart
                    if backtest_results['equity_curve']:
                        st.markdown("### 📊 Equity Curve")
                        
                        equity_df = pd.DataFrame(backtest_results['equity_curve'])
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(equity_df['Date'], equity_df['Portfolio_Value'], 
                               color='#1E88E5', linewidth=2, label='Portfolio Value')
                        
                        # Add buy & hold comparison
                        initial_shares = backtest_capital / float(df['Close'].iloc[0])
                        buy_hold_values = [initial_shares * price for price in equity_df['Price']]
                        ax.plot(equity_df['Date'], buy_hold_values, 
                               color='#FF6B6B', linewidth=2, linestyle='--', label='Buy & Hold')
                        
                        ax.set_title(f'{selected_strategy_name} Strategy Performance')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Portfolio Value ($)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                          # Format y-axis as currency
                        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        
                        st.pyplot(fig)
                    
                    # Detailed metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Performance Metrics")
                        st.markdown(f"""
                        - **Final Capital:** ${backtest_results['final_capital']:,.2f}
                        - **Profit Factor:** {backtest_results.get('profit_factor', 'N/A')}
                        - **Average Win:** ${backtest_results.get('avg_win', 0):.2f}
                        - **Average Loss:** ${abs(backtest_results.get('avg_loss', 0)):.2f}
                        - **Max Drawdown:** {backtest_results.get('max_drawdown_pct', 0):.1f}%
                        """)
                    
                    with col2:
                        st.markdown("### 🔍 Trade Analysis")
                        if backtest_results['trades']:
                            trade_df = pd.DataFrame(backtest_results['trades'])
                            exit_trades = trade_df[trade_df['Type'] == 'Exit']
                            
                            if not exit_trades.empty:
                                avg_hold_time = exit_trades['Days_Held'].mean()
                                best_trade = exit_trades['Profit_Loss_Pct'].max() * 100
                                worst_trade = exit_trades['Profit_Loss_Pct'].min() * 100
                                
                                st.markdown(f"""
                                - **Average Hold Time:** {avg_hold_time:.1f} days
                                - **Best Trade:** {best_trade:.1f}%
                                - **Worst Trade:** {worst_trade:.1f}%
                                - **Profitable Trades:** {len(exit_trades[exit_trades['Profit_Loss'] > 0])}
                                - **Losing Trades:** {len(exit_trades[exit_trades['Profit_Loss'] <= 0])}
                                """)
                    
                    # Trade history table
                    if backtest_results['trades']:
                        st.markdown("### 📋 Recent Trades")
                        trade_df = pd.DataFrame(backtest_results['trades'])
                        exit_trades = trade_df[trade_df['Type'] == 'Exit'].tail(10)
                        
                        if not exit_trades.empty:
                            display_trades = exit_trades[['Date', 'Entry_Date', 'Entry_Price', 'Price', 'Profit_Loss_Pct', 'Exit_Reason']].copy()
                            display_trades['Profit_Loss_Pct'] = (display_trades['Profit_Loss_Pct'] * 100).round(2)
                            display_trades['Entry_Price'] = display_trades['Entry_Price'].round(2)
                            display_trades['Price'] = display_trades['Price'].round(2)
                            display_trades.columns = ['Exit Date', 'Entry Date', 'Entry Price', 'Exit Price', 'Return (%)', 'Exit Reason']
                            
                            st.dataframe(display_trades, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
                    st.info("This might be due to insufficient data or issues with the technical indicators. Try a different ticker or time period.")
        
        # Custom strategy builder section
        st.markdown("---")
        st.markdown("### 🛠️ Custom Strategy Builder")
        
        with st.expander("Build Your Own Strategy", expanded=False):
            st.markdown("**Coming Soon:** Interactive strategy builder with custom entry/exit conditions")
            st.info("""
            Future features will include:
            - Drag-and-drop condition builder
            - Custom indicator combinations  
            - Advanced risk management rules
            - Multi-timeframe analysis
            - Strategy optimization
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Initial empty state when the app first loads
    st.markdown("""
    <div style='text-align: center; padding: 50px 0;'>
        <img src="https://static.streamlit.io/examples/stock.svg" style="width: 150px; margin-bottom: 20px;">
        <h2>Welcome to Swing Trade Pro</h2>
        <p style='color: #666; margin-bottom: 30px;'>Enter a stock ticker in the sidebar and click "Run Analysis" to get started.</p>
    </div>
    """, unsafe_allow_html=True)