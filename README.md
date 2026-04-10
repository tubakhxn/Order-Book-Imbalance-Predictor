# Order Book Imbalance Price Predictor

## Developer
**tubakhxn**

## Project Overview
This project simulates or processes Level 2 order book data and predicts short-term price movement using machine learning. Inspired by high-frequency trading (HFT) strategies, it leverages microstructure signals such as order book imbalance, bid-ask spread, and order flow to forecast the next price direction (up/down). The project includes strong visual insights with heatmaps, time-series, and feature importance plots.

## Features
- Simulates order book data (bids, asks, volumes)
- Feature engineering: bid-ask spread, order book imbalance, mid price, volatility, order flow imbalance
- Machine learning model: RandomForest or XGBoost classifier
- Visualizations: order book heatmap, price vs prediction, confidence plot, feature importance
- Outputs: accuracy, confusion matrix, feature importances

## How to Fork and Run
1. **Fork this repository** on GitHub (click the "Fork" button at the top right).
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Order-Book-Imbalance-Predictor.git
   ```
3. **Navigate to the project directory**:
   ```bash
   cd Order-Book-Imbalance-Predictor
   ```
4. **(Optional) Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Run the project**:
   ```bash
   python main.py
   ```

## Relevant Wikipedia Links
- [Order book (finance)](https://en.wikipedia.org/wiki/Order_book_(trading))
- [Bid–ask spread](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread)
- [Order flow](https://en.wikipedia.org/wiki/Order_flow_trading)
- [High-frequency trading](https://en.wikipedia.org/wiki/High-frequency_trading)
- [Random forest](https://en.wikipedia.org/wiki/Random_forest)
- [XGBoost](https://en.wikipedia.org/wiki/XGBoost)

---

*Created by tubakhxn, 2026*