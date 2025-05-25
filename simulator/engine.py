import yfinance as yf
import pandas as pd

from indicators.rsi import calculate_rsi
from indicators.sma import add_sma_signals
from strategies.ma_crossover import check_entry_signal
from risk.position_sizing import get_position_size
from journal.trade_logger import log_trade
from simulator.plot_trade import plot_trade

def run_simulation(ticker, capital):
    print(f"Starting simulation for {ticker} with ${capital} capital...\n")

    df = yf.download(tickers=ticker, period="6mo", interval="1d", group_by='ticker')
    df = df[ticker].copy()  # ✅ Gets you just the OHLCV for that ticker

    print("Clean columns:", df.columns)
    print("Data shape:", df.shape)
    

    df = calculate_rsi(df)
    df = add_sma_signals(df)
    
    print("SMA20 preview:", df['SMA20'].dropna().tail())
    print("SMA50 preview:", df['SMA50'].dropna().tail())

 
 
    for i in range(20, len(df) - 10):  # ensure loop has enough data points
        if i % 5 == 0:
            print(f"Scanning {df.index[i].date()}...")  # progress every 5 steps


        if check_entry_signal(df.iloc[:i]):
            print(f"✅ Signal detected on {df.index[i].date()}")

            entry_index = i
            exit_index = i + 10

            entry_price = float(df['Close'].iloc[entry_index])
            exit_price = float(df['Close'].iloc[exit_index])
            shares = get_position_size(capital, entry_price)

            profit = round((exit_price - entry_price) * shares, 2)
            return_pct = round((exit_price - entry_price) / entry_price * 100, 2)

            trade = {
                "ticker": ticker,
                "entry_date": str(df.index[entry_index].date()),
                "entry_price": round(entry_price, 2),
                "exit_date": str(df.index[exit_index].date()),
                "exit_price": round(exit_price, 2),
                "shares": shares,
                "profit": profit,
                "return_pct": return_pct
            }

            log_trade(trade)
            plot_trade(df, entry_index, ticker)
            print(f"""
                📈 TRADE EXECUTED: {trade['ticker']}
                Entry Date: {trade['entry_date']} @ ${trade['entry_price']}
                Exit Date : {trade['exit_date']} @ ${trade['exit_price']}
                Shares    : {trade['shares']}
                PnL       : ${trade['profit']} ({trade['return_pct']}%)
""")




