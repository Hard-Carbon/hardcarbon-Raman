from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def _vif_for_column(values: np.ndarray, col_idx: int) -> float:
    y = values[:, col_idx]
    if np.nanvar(y) <= 1e-14:
        return math.inf
    other_idx = [i for i in range(values.shape[1]) if i != col_idx]
    if not other_idx:
        return 1.0
    X_other = values[:, other_idx]
    try:
        model = LinearRegression()
        model.fit(X_other, y)
        r2 = model.score(X_other, y)
    except Exception:
        return math.inf
    r2 = min(max(float(r2), 0.0), 1.0)
    denom = 1.0 - r2
    if denom <= 1e-12:
        return math.inf
    return 1.0 / denom

def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] == 0:
        return pd.DataFrame(columns=['feature', 'vif'])
    if X.shape[1] == 1:
        return pd.DataFrame({'feature': X.columns, 'vif': [1.0]})
    values = X.astype(float).to_numpy()
    rows = []
    for idx, feature in enumerate(X.columns):
        rows.append({'feature': feature, 'vif': _vif_for_column(values, idx)})
    return pd.DataFrame(rows).sort_values('vif', ascending=False, na_position='first')

def iterative_vif_filter(X_train: pd.DataFrame, threshold: float=10.0, min_features: int=1) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = list(X_train.columns)
    before = compute_vif_table(X_train[current])
    removed_rows = []
    iteration = 0
    while len(current) > max(1, min_features):
        vif_table = compute_vif_table(X_train[current])
        if vif_table.empty:
            break
        top = vif_table.iloc[0]
        top_vif = float(top['vif']) if np.isfinite(top['vif']) else math.inf
        if top_vif <= threshold:
            break
        feature = str(top['feature'])
        current.remove(feature)
        removed_rows.append({'iteration': iteration, 'removed_feature': feature, 'removed_feature_vif': top_vif, 'remaining_feature_count': len(current)})
        iteration += 1
    after = compute_vif_table(X_train[current])
    removed = pd.DataFrame(removed_rows)
    return (current, before, after, removed)
