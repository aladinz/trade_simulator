def get_position_size(capital, entry_price, risk_pct=0.01):
    risk_amount = capital * risk_pct
    if hasattr(entry_price, 'iloc'):
        entry_price = entry_price.iloc[0]  # safely extract the scalar
    return int(risk_amount / entry_price)

