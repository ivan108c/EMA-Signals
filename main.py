import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# EXTRACTING THE DATA

df = yf.download('BTC-USD', start='2023-06-20', end='2024-07-04', interval='1D')
closing_price = df['Adj Close']
print(closing_price.to_string())

# EMA 

window_size = 100
smoothing_factor = 2 / (window_size + 1)
initial_ema = []
initial_ema.append(closing_price[0])

for i in range(1, len(closing_price)):
    new_ema = ((closing_price[i] * smoothing_factor) + (initial_ema[-1] * (1 - smoothing_factor)))
    initial_ema.append(new_ema)

df["EMA"] = initial_ema

# BUY and SELL signals

buy_signals = []
sell_signals = []

chunk_size = 5  
signal_period = 10

last_buy_signal = None
last_sell_signal = None

for i in range(window_size, len(closing_price) - chunk_size + 1):
    if last_buy_signal is None or (df.index[i] - last_buy_signal).days >= signal_period:
        if np.all(closing_price.iloc[i:i + chunk_size] > df['EMA'].iloc[i:i + chunk_size]):
            buy_signals.append((df.index[i + chunk_size - 1], closing_price.iloc[i + chunk_size - 1]))
            last_buy_signal = df.index[i + chunk_size - 1]

    if last_sell_signal is None or (df.index[i] - last_sell_signal).days >= signal_period:
        if np.all(closing_price.iloc[i:i + chunk_size] < df['EMA'].iloc[i:i + chunk_size]):
            sell_signals.append((df.index[i + chunk_size - 1], closing_price.iloc[i + chunk_size - 1]))
            last_sell_signal = df.index[i + chunk_size - 1]

buy_dates, buy_prices = zip(*buy_signals)
sell_dates, sell_prices = zip(*sell_signals)

# PLOTTING THE DATA

plt.figure(figsize=(14, 7))
plt.title("BTC-USD")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(df.index, df['Adj Close'], label="Adjusted Close", color="Blue")
plt.plot(df.index, df['EMA'], label="EMA", color="Red")
plt.scatter(buy_dates, [price + 10 for price in buy_prices], marker='^', color='green', label='Buy Signal', s=100, alpha=1.0)
plt.scatter(sell_dates, [price - 10 for price in sell_prices], marker='v', color='red', label='Sell Signal', s=100, alpha=1.0)
plt.legend()
plt.show()
