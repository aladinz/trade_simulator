def add_sma_signals(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    return df
