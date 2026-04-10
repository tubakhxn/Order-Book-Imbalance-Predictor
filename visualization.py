import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_order_book_heatmap(df, n_levels=5, n_steps=100):
    bids = np.array(df['bid_volumes'].tolist())[:n_steps, :n_levels]
    asks = np.array(df['ask_volumes'].tolist())[:n_steps, :n_levels]
    return bids, asks

def plot_price_vs_prediction(df, y_pred):
    return df['mid_price'], y_pred, df.index

def plot_confidence(y_proba):
    return y_proba

def plot_feature_importance(importances, feature_names):
    return importances, feature_names

def plot_all_together(df, y_pred, y_proba, importances, feature_names, n_levels=5, n_steps=100):
    """
    Plot all graphs together in a single figure with subplots.
    """
    bids, asks = np.array(df['bid_volumes'].tolist())[:n_steps, :n_levels], np.array(df['ask_volumes'].tolist())[:n_steps, :n_levels]
    mid_price = df['mid_price'].iloc[-n_steps:]
    idx = df.index[-n_steps:]
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    # Heatmaps
    sns.heatmap(bids.T, cmap='Blues', ax=axs[0, 0], cbar=True)
    axs[0, 0].set_title('Bid Depth (Heatmap)')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Level')
    sns.heatmap(asks.T, cmap='Reds', ax=axs[0, 1], cbar=True)
    axs[0, 1].set_title('Ask Depth (Heatmap)')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Level')
    # Price vs Prediction
    axs[1, 0].plot(mid_price, label='Mid Price')
    if y_pred is not None:
        axs[1, 0].scatter(idx, mid_price, c=y_pred[-n_steps:], cmap='coolwarm', label='Predicted Direction', alpha=0.5)
    axs[1, 0].set_title('Mid Price vs Predicted Direction')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Price')
    axs[1, 0].legend()
    # Confidence or Feature Importance
    if y_proba is not None:
        axs[1, 1].plot(y_proba[-n_steps:], label='Predicted Probability (Up)')
        axs[1, 1].set_title('Prediction Confidence Over Time')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Probability')
        axs[1, 1].legend()
    elif importances is not None:
        sns.barplot(x=importances, y=feature_names, ax=axs[1, 1])
        axs[1, 1].set_title('Feature Importance')
        axs[1, 1].set_xlabel('Importance')
        axs[1, 1].set_ylabel('Feature')
    plt.tight_layout()
    plt.show()
