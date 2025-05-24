# simulator/plot_trade.py

import mplfinance as mpf
import pandas as pd

def plot_trade(df, entry_index, ticker):
    df = df.copy()
    df.index.name = 'Date'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50']]

    # Focus on 60 bars around the trade
    start = max(entry_index - 30, 0)
    end = min(entry_index + 30, len(df))
    plot_df = df.iloc[start:end]

    # Create overlays for SMAs
    apds = [
        mpf.make_addplot(plot_df['SMA20'], color='blue'),
        mpf.make_addplot(plot_df['SMA50'], color='orange'),
    ]

    # Entry line
    entry_date = df.index[entry_index]
    low = float(plot_df['Low'].min())
    high = float(plot_df['High'].max())

    alines = dict(
    alines=[[(entry_date, low), (entry_date, high)]],
    colors=['green'],
    linewidths=[1.5]
)


    mpf.plot(
        plot_df,
        type='candle',
        addplot=apds,
        alines=alines,
        style='yahoo',
        title=f"Simulated Trade Entry: {ticker} on {entry_date.date()}",
        ylabel="Price",
        volume=True,
        warn_too_much_data=1000
    )
