from __future__ import annotations
import numpy as np
import pandas as pd

def _rank_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    n_valid = values.notna().sum()
    if n_valid <= 1:
        return pd.Series(np.where(values.notna(), 1.0, 0.0), index=series.index)
    ranks = values.rank(ascending=not higher_is_better, method='average', na_option='bottom')
    scores = 1.0 - (ranks - 1.0) / (n_valid - 1.0)
    scores[values.isna()] = 0.0
    return scores.clip(lower=0.0, upper=1.0)
