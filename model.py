import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

def train_model(X_train, y_train, model_type='rf', random_state=42):
    if model_type == 'xgb' and xgb_available:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = None
    # For probability/confidence plot
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None
    return acc, cm, importances, y_pred, y_proba
