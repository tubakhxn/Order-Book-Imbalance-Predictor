import numpy as np
import pandas as pd

def simulate_order_book(n_steps=1000, n_levels=5, seed=42):
    np.random.seed(seed)
    mid_price = 100.0
    prices = [mid_price]
    bid_prices, ask_prices = [], []
    bid_volumes, ask_volumes = [], []
    for _ in range(n_steps):
        spread = np.round(np.random.uniform(0.01, 0.05), 2)
        mid = prices[-1] + np.random.normal(0, 0.02)
        prices.append(mid)
        bids = np.sort(mid - np.abs(np.random.uniform(0.01, 0.2, n_levels)))[::-1]
        asks = np.sort(mid + np.abs(np.random.uniform(0.01, 0.2, n_levels)))
        bid_vol = np.random.randint(10, 100, n_levels)
        ask_vol = np.random.randint(10, 100, n_levels)
        bid_prices.append(bids)
        ask_prices.append(asks)
        bid_volumes.append(bid_vol)
        ask_volumes.append(ask_vol)
    df = pd.DataFrame({
        'mid_price': prices[1:],
        'bid_prices': bid_prices,
        'ask_prices': ask_prices,
        'bid_volumes': bid_volumes,
        'ask_volumes': ask_volumes
    })
    return df

def load_data(simulate=True, **kwargs):
    if simulate:
        return simulate_order_book(**kwargs)
    else:
        raise NotImplementedError('Only simulation is implemented.')
