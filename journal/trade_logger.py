import csv
import os

def log_trade(trade, file_path="journal/trade_log.csv"):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=trade.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)

