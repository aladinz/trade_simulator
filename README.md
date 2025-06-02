# 📈 Swing Trade Pro - Advanced Trading Analytics Platform

## Overview

**Swing Trade Pro** is a comprehensive stock trading analysis and backtesting platform built with Streamlit. It combines technical analysis, machine learning predictions, sentiment analysis, and advanced strategy backtesting to provide traders with data-driven insights for swing trading decisions.

## 🚀 Features

### 📊 Market Analysis
- **Real-time Stock Data**: Live market data via Yahoo Finance API
- **Technical Indicators**: SMA (20/50), RSI, Support/Resistance levels
- **Price Charts**: Interactive candlestick charts with technical overlays
- **Market Sentiment**: Trend analysis and trading signals

### 🧠 Machine Learning Insights
- **Ensemble Prediction Model**: Random Forest + XGBoost for 90-day forecasts
- **Feature Importance Analysis**: 16 technical and time-based indicators
- **Model Performance Metrics**: RMSE, MAE with 95% confidence intervals
- **Prediction Visualization**: Interactive charts with confidence bands

### 📰 News & Sentiment Analysis
- **Real-time News Integration**: Latest market news and headlines
- **Sentiment Scoring**: Automated sentiment analysis of news articles
- **Market Impact Assessment**: News-driven market sentiment indicators

### 📈 Strategy Backtesting
- **Pre-built Strategies**:
  - RSI Mean Reversion
  - SMA Crossover
  - Bollinger Band Breakout
  - Momentum Combo
- **Performance Metrics**: Win rate, total return, Sharpe ratio, max drawdown
- **Equity Curve Analysis**: Portfolio value tracking vs buy-and-hold
- **Trade Analysis**: Detailed trade history and statistics
- **Risk Management**: Customizable stop-loss and position sizing

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd trade_sim
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows PowerShell
   # OR
   source .venv/bin/activate   # On Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the App**
   - Open your browser to `http://localhost:8501`
   - Enter a stock ticker (e.g., "AMD", "AAPL", "TSLA")
   - Click "🚀 Run Analysis"

## 📋 Requirements

```
streamlit>=1.28.0
yfinance>=0.2.18
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
plotly>=5.15.0
mplfinance>=0.12.9
scikit-learn>=1.3.0
xgboost>=1.7.0
requests>=2.31.0
```

## 🏗️ Project Structure

```
trade_sim/
├── app.py                          # Main Streamlit application
├── trading_strategies.py           # Strategy backtesting engine
├── predictor_fixed.py             # ML prediction models
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── indicators/                    # Technical indicators
│   ├── __init__.py
│   ├── sma.py                    # Simple Moving Average
│   ├── rsi.py                    # Relative Strength Index
│   └── advanced_indicators.py    # Additional indicators
│
├── strategies/                    # Trading strategies
│   └── ma_crossover.py           # Moving average crossover
│
├── simulator/                     # Trading simulation
│   ├── engine.py                 # Simulation engine
│   └── plot_trade.py            # Trade visualization
│
├── risk/                         # Risk management
│   └── position_sizing.py       # Position sizing algorithms
│
├── journal/                      # Trade logging
│   ├── trade_logger.py          # Trade logging functionality
│   └── trade_log.csv            # Trade history
│
├── sentiment/                    # Sentiment analysis
│   └── sentiment_engine.py      # News sentiment analysis
│
└── data/                         # Data storage
    └── README.md                # Data directory info
```

## 🎯 Usage Guide

### 1. Basic Stock Analysis
1. Enter a stock ticker symbol (e.g., "AMD")
2. Set your trading capital
3. Click "🚀 Run Analysis"
4. Review the **Market Analysis** tab for current metrics

### 2. ML Predictions
1. Navigate to the **ML Insights** tab
2. View 90-day price forecasts
3. Analyze feature importance for prediction models
4. Review model performance metrics

### 3. Strategy Backtesting
1. Go to the **Strategy Backtesting** tab
2. Select a pre-built strategy:
   - **RSI Mean Reversion**: Buy oversold, sell overbought
   - **SMA Crossover**: Trend-following strategy
   - **Bollinger Band Breakout**: Volatility-based strategy
   - **Momentum Combo**: Multi-indicator approach
3. Configure risk parameters (optional)
4. Set backtest capital amount
5. Click "🚀 Run Backtest"
6. Analyze results: win rate, total return, trade history

### 4. News Sentiment
- Check the **News & Sentiment** tab for recent market news
- Review sentiment scores and market impact indicators

## 📊 Key Metrics Explained

### Strategy Performance
- **Total Return**: Overall percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Ratio of gross profit to gross loss

### ML Model Metrics
- **RMSE**: Root Mean Square Error - prediction accuracy
- **MAE**: Mean Absolute Error - average prediction error
- **Confidence Level**: Statistical confidence in predictions

## 🔧 Configuration

### Risk Management
- **Stop Loss**: Maximum acceptable loss per trade (1-15%)
- **Take Profit**: Target profit level (5-30%)
- **Risk per Trade**: Portfolio risk allocation (1-5%)

### Position Sizing
- Automatic calculation based on:
  - Portfolio value
  - Risk tolerance
  - Current stock price
  - Stop loss level

## 🐛 Troubleshooting

### Common Issues

1. **"Invalid or insufficient data" Error**
   - Try a different ticker symbol
   - Ensure the stock has sufficient trading history (6+ months)

2. **"Error running backtest" Message**
   - Check if the stock has enough historical data
   - Try a different time period or ticker

3. **Blank Charts**
   - Refresh the page and re-run analysis
   - Check internet connection for data fetching

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment

### Debug Mode
If you encounter issues, you can:
- Check the error messages in the Streamlit interface
- Review the troubleshooting section above
- Ensure all dependencies are properly installed

## 🔄 Recent Updates

### Version 2.0.0 (Latest)
✅ **Fixed Strategy Backtesting Issues**
- Resolved FuncFormatter import conflict
- Fixed Total Trades display visibility
- Corrected syntax errors in metric display
- Validated complete workflow functionality

### Features Added
- Enhanced error handling and validation
- Improved UI styling and responsiveness
- Better metric visualization with color coding
- Streamlined project structure and cleanup

## 🚀 Future Enhancements

### Planned Features
- **Custom Strategy Builder**: Drag-and-drop condition builder
- **Multi-timeframe Analysis**: Multiple chart timeframes
- **Portfolio Management**: Multi-stock portfolio tracking
- **Advanced Risk Models**: VaR, CVaR calculations
- **Paper Trading**: Live simulation mode
- **Export Functionality**: Strategy and trade reports

### API Integrations
- Alpha Vantage for extended data
- Polygon.io for real-time quotes
- Twitter sentiment analysis
- Economic calendar integration

## 📖 Documentation

### Core Modules

1. **app.py**: Main Streamlit interface with four tabs
2. **trading_strategies.py**: Strategy engine with backtesting
3. **predictor_fixed.py**: ML models for price prediction
4. **indicators/**: Technical analysis calculations
5. **simulator/**: Trade execution and visualization

### Strategy Development
To create custom strategies:
1. Extend the `TradingStrategy` class
2. Define entry/exit conditions
3. Implement risk management rules
4. Add to strategy options in `app.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed description

---

**Swing Trade Pro** - Empowering traders with data-driven insights and advanced analytics.

*Last Updated: June 2, 2025*