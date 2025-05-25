
# This script implements a simple moving average (SMA) crossover strategy.
# It checks for entry signals based on the crossover of two SMAs (20 and 50 periods).
# It also includes a function to add SMA signals to a DataFrame and a function

def check_entry_signal(df):
    return (
        df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and
        df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2]
    )


# This function checks if the last closing price is above the last SMA20 and
# if the last closing price is below the last SMA50. If both conditions are met,
# it returns True, indicating a potential entry signal. Otherwise, it returns False.


