"""
Advanced Technical Indicators for Swing Trade Pro

This module provides advanced technical analysis indicators for stock trading:
- Volume Profile indicator
- Ichimoku Cloud indicator
- Pivot Points indicator
- Heikin-Ashi candles
- Average True Range (ATR) indicator
- On-Balance Volume (OBV) indicator
"""
import pandas as pd
import numpy as np


def calculate_volume_profile(df, bins=10):
    """
    Calculate volume profile to identify price levels with high trading volumes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Date, Open, High, Low, Close, Volume columns
    bins : int
        Number of price bins for volume distribution
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with volume profile data added
    """
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Get price range for the period
    price_min = df['Low'].min()
    price_max = df['High'].max()
    
    # Create price bins
    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    # Initialize volume array for each price bin
    volume_profile = np.zeros(bins)
    
    # Distribute volume across price bins for each day
    for i, row in df.iterrows():
        # Get the high and low price for this day
        low = row['Low']
        high = row['High']
        volume = row['Volume']
        
        # Calculate the portion of the price range that falls into each bin
        for j in range(bins):
            bin_low = price_bins[j]
            bin_high = price_bins[j + 1]
            
            # Calculate overlap between day's price range and bin
            overlap_low = max(low, bin_low)
            overlap_high = min(high, bin_high)
            
            # If there's an overlap, distribute volume proportional to the overlap
            if overlap_high > overlap_low:
                overlap_ratio = (overlap_high - overlap_low) / (high - low)
                volume_profile[j] += volume * overlap_ratio
    
    # Add volume profile data to dataframe
    result_df['volume_profile_bins'] = [bin_centers.tolist()] * len(df)
    result_df['volume_profile_values'] = [volume_profile.tolist()] * len(df)    # Find high volume nodes (potential support/resistance)
    high_volume_threshold = np.percentile(volume_profile, 75)
    # Create a boolean mask and then use it to get high volume bins
    mask = volume_profile > high_volume_threshold
    high_volume_bins = bin_centers[mask]
    result_df['volume_profile_poc'] = [high_volume_bins.tolist() if len(high_volume_bins) > 0 else []] * len(df)
    
    return result_df


def calculate_ichimoku_cloud(df):
    """
    Calculate Ichimoku Cloud indicator.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Date, Open, High, Low, Close, Volume columns
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with Ichimoku Cloud components added
    """
    # Check if we have enough data points for calculations
    if len(df) < 55:  # Need at least 52 + 3 periods for minimal calculation
        raise ValueError("Not enough data points for Ichimoku Cloud calculation (min 55 required)")
        
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    result_df['ichimoku_tenkan_sen'] = (high_9 + low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    result_df['ichimoku_kijun_sen'] = (high_26 + low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2, shifted 26 periods ahead
    result_df['ichimoku_senkou_span_a'] = ((result_df['ichimoku_tenkan_sen'] + 
                                          result_df['ichimoku_kijun_sen']) / 2).shift(26)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 periods ahead
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    result_df['ichimoku_senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    # Calculate Chikou Span (Lagging Span): Close price, shifted 26 periods back
    result_df['ichimoku_chikou_span'] = result_df['Close'].shift(-26)
    
    return result_df


def calculate_pivot_points(df):
    """
    Calculate pivot points for price support and resistance levels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Date, Open, High, Low, Close, Volume columns
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with pivot point data added
    """
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Group the data by date to get daily OHLC
    # Note: If data is already daily, this step can be skipped
    # For this function we'll assume data is already daily
    
    # Initialize pivot columns
    result_df['pivot_point'] = pd.NA
    result_df['pivot_r1'] = pd.NA
    result_df['pivot_r2'] = pd.NA
    result_df['pivot_r3'] = pd.NA
    result_df['pivot_s1'] = pd.NA
    result_df['pivot_s2'] = pd.NA
    result_df['pivot_s3'] = pd.NA
    
    # Calculate pivot points for each day based on previous day's data
    for i in range(1, len(result_df)):
        # Get previous day's data
        prev_high = result_df['High'].iloc[i-1]
        prev_low = result_df['Low'].iloc[i-1]
        prev_close = result_df['Close'].iloc[i-1]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = pivot + 2 * (prev_high - prev_low)
        s3 = pivot - 2 * (prev_high - prev_low)
        
        # Store in DataFrame
        result_df.loc[result_df.index[i], 'pivot_point'] = pivot
        result_df.loc[result_df.index[i], 'pivot_r1'] = r1
        result_df.loc[result_df.index[i], 'pivot_r2'] = r2
        result_df.loc[result_df.index[i], 'pivot_r3'] = r3
        result_df.loc[result_df.index[i], 'pivot_s1'] = s1
        result_df.loc[result_df.index[i], 'pivot_s2'] = s2
        result_df.loc[result_df.index[i], 'pivot_s3'] = s3
    
    return result_df


def calculate_heikin_ashi(df):
    """
    Calculate Heikin-Ashi candles for smoother price action visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Date, Open, High, Low, Close, Volume columns
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with Heikin-Ashi data added
    """
    # Check if we have enough data
    if len(df) < 2:
        raise ValueError("Not enough data points for Heikin-Ashi calculation (min 2 required)")
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate Heikin-Ashi candles
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize HA Open with first day's Open value
    ha_open = [df['Open'].iloc[0]]
    
    # Calculate HA Open for each day after the first one
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)
    
    # Create Series for HA Open
    ha_open = pd.Series(ha_open, index=df.index)
    
    # Calculate HA High and Low
    ha_high = pd.Series([max(h, o, c) for h, o, c in zip(df['High'], ha_open, ha_close)], index=df.index)
    ha_low = pd.Series([min(l, o, c) for l, o, c in zip(df['Low'], ha_open, ha_close)], index=df.index)
      # Add Heikin-Ashi data to result DataFrame
    result_df['ha_open'] = ha_open
    result_df['ha_high'] = ha_high
    result_df['ha_low'] = ha_low
    result_df['ha_close'] = ha_close
    
    # Calculate Heikin-Ashi trend using .values to avoid Series ambiguity
    result_df['ha_trend'] = 'neutral'
    
    # Convert Series to arrays for boolean comparison
    ha_close_values = ha_close.values
    ha_open_values = ha_open.values
    
    bullish_mask = ha_close_values > ha_open_values
    bearish_mask = ha_close_values < ha_open_values
    
    result_df.loc[bullish_mask, 'ha_trend'] = 'bullish'
    result_df.loc[bearish_mask, 'ha_trend'] = 'bearish'
    
    return result_df


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Date, Open, High, Low, Close, Volume columns
    period : int
        Number of periods to use for ATR calculation (default: 14)
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with ATR data added
    """
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close_prev = abs(df['High'] - df['Close'].shift(1))
    low_close_prev = abs(df['Low'] - df['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR
    result_df['ATR'] = true_range.rolling(window=period).mean()
    
    # Calculate Normalized ATR (ATR as a percentage of close price)
    result_df['ATR_percent'] = result_df['ATR'] / result_df['Close'] * 100
    
    # Calculate volatility bands (for potential stop-loss placement)
    result_df['ATR_upper'] = result_df['Close'] + (result_df['ATR'] * 2)
    result_df['ATR_lower'] = result_df['Close'] - (result_df['ATR'] * 2)
    
    return result_df


def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV) indicator for volume-price analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Date, Open, High, Low, Close, Volume columns
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with OBV data added
    """
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate OBV
    obv = [0]  # Initialize with 0
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    # Add OBV to result dataframe
    result_df['OBV'] = obv
    
    # Calculate OBV EMA for signal line
    result_df['OBV_EMA'] = result_df['OBV'].ewm(span=20, adjust=False).mean()
    
    # Check for OBV divergence
    # Price making new highs but OBV not confirming = bearish
    # Price making new lows but OBV not confirming = bullish
    result_df['OBV_signal'] = 'neutral'
    
    for i in range(20, len(result_df)):
        # Get recent price and OBV data
        price_window = result_df['Close'].iloc[i-20:i+1]
        obv_window = result_df['OBV'].iloc[i-20:i+1]        # Check if price made new high but OBV didn't - using scalar values
        price_current = float(price_window.iloc[-1])
        price_max = float(price_window.max())
        obv_current = float(obv_window.iloc[-1])
        obv_max = float(obv_window.max())
        
        price_at_max = abs(price_current - price_max) < 0.0001  # Floating point comparison
        obv_not_at_max = abs(obv_current - obv_max) >= 0.0001   # Floating point comparison
        if price_at_max and obv_not_at_max:
            result_df.loc[result_df.index[i], 'OBV_signal'] = 'bearish_divergence'        # Check if price made new low but OBV didn't - using scalar values
        price_min = float(price_window.min())
        obv_min = float(obv_window.min())
        
        price_at_min = abs(price_current - price_min) < 0.0001  # Floating point comparison
        obv_not_at_min = abs(obv_current - obv_min) >= 0.0001   # Floating point comparison
        if price_at_min and obv_not_at_min:
            result_df.loc[result_df.index[i], 'OBV_signal'] = 'bullish_divergence'
    
    return result_df


# Self-test code to run when module is executed directly
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    
    print("Advanced Indicators Module - Self Test")
    print("======================================")
    
    # Create test data or download real data
    try:
        print("Downloading test data...")
        df = yf.download("MSFT", period="3mo")
        df.reset_index(inplace=True)
        print(f"Downloaded {len(df)} rows of market data")
    except:
        print("Creating synthetic test data...")
        # Create synthetic data if download fails
        dates = pd.date_range(start="2024-01-01", periods=100)
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.normal(0, 1, 100))
        high = close + np.random.uniform(0, 3, 100)
        low = close - np.random.uniform(0, 3, 100)
        open_price = low + np.random.uniform(0, high-low, 100)
        volume = np.random.uniform(1000000, 10000000, 100)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        print("Created synthetic data with 100 rows")
    
    # Test each indicator
    indicators = [
        ("Volume Profile", calculate_volume_profile),
        ("Ichimoku Cloud", calculate_ichimoku_cloud),
        ("Pivot Points", calculate_pivot_points),
        ("Heikin-Ashi", calculate_heikin_ashi),
        ("ATR", calculate_atr),
        ("OBV", calculate_obv)
    ]
    
    all_passed = True
    
    for name, func in indicators:
        print(f"\nTesting {name}...")
        try:
            result = func(df)
            new_cols = set(result.columns) - set(df.columns)
            print(f"✓ SUCCESS - Added {len(new_cols)} columns: {', '.join(sorted(new_cols))}")
            
            # Print sample data for first new column
            if new_cols:
                sample_col = sorted(new_cols)[0]
                print(f"Sample values for {sample_col}:")
                print(result[sample_col].tail(3))
        except Exception as e:
            print(f"✗ ERROR - {str(e)}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    if all_passed:
        print("\n✓ All indicators tested successfully!")
        print("Advanced Indicators module is ready for use in the app.")
    else:
        print("\n✗ Some indicators had errors. Please fix the issues before using in the app.")
