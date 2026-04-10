import numpy as np
import pandas as pd

def compute_features(df, n_levels=5, window=20):
    features = pd.DataFrame(index=df.index)
    # Bid-ask spread (best level)
    features['spread'] = df['ask_prices'].apply(lambda x: x[0]) - df['bid_prices'].apply(lambda x: x[0])
    # Order book imbalance (best level)
    features['obi'] = (
        df['bid_volumes'].apply(lambda x: x[0]) - df['ask_volumes'].apply(lambda x: x[0])
    ) / (
        df['bid_volumes'].apply(lambda x: x[0]) + df['ask_volumes'].apply(lambda x: x[0])
    )
    # Mid price
    features['mid_price'] = df['mid_price']
    # Rolling volatility
    features['volatility'] = df['mid_price'].rolling(window).std().fillna(0)
    # Order flow imbalance (sum of bid - ask volumes at all levels)
    features['ofi'] = (
        df['bid_volumes'].apply(np.sum) - df['ask_volumes'].apply(np.sum)
    ) / (
        df['bid_volumes'].apply(np.sum) + df['ask_volumes'].apply(np.sum)
    )
    return features

def compute_target(df, horizon=1):
    # Next price direction: 1 if up, 0 if down or unchanged
    future_price = df['mid_price'].shift(-horizon)
    return (future_price > df['mid_price']).astype(int)
