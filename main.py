import numpy as np
import pandas as pd
from data_loader import load_data
from feature_engineering import compute_features, compute_target
from model import train_model, evaluate_model
from visualization import (
    plot_all_together
)
from sklearn.model_selection import train_test_split

# 1. Load/simulate data
print('Simulating order book data...')
df = load_data(simulate=True, n_steps=1000, n_levels=5)

# 2. Feature engineering
print('Computing features...')
features = compute_features(df, n_levels=5)
target = compute_target(df, horizon=1)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 4. Train model
print('Training model...')
model = train_model(X_train, y_train, model_type='rf')

# 5. Evaluate
print('Evaluating model...')
acc, cm, importances, y_pred, y_proba = evaluate_model(model, X_test, y_test)
print(f'Accuracy: {acc:.4f}')
print('Confusion Matrix:')
print(cm)

# 6. Visualizations (all together)
plot_all_together(
    df,
    y_pred,
    y_proba,
    importances,
    features.columns,
    n_levels=5,
    n_steps=100
)
