from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

def feature_effect_direction(feature_values: np.ndarray, shap_values: np.ndarray) -> float:
    x = np.asarray(feature_values, dtype=float)
    s = np.asarray(shap_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(s)
    if finite.sum() < 3 or np.std(x[finite]) <= 1e-12 or np.std(s[finite]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x[finite], s[finite])[0, 1])

def _pairwise_repeat_agreement(repeat_table: pd.DataFrame, top_k: int) -> dict[str, float]:
    repeats = sorted(repeat_table['repeat'].unique())
    jaccards: list[float] = []
    taus: list[float] = []
    for left, right in combinations(repeats, 2):
        a = repeat_table[repeat_table['repeat'].eq(left)].set_index('feature')
        b = repeat_table[repeat_table['repeat'].eq(right)].set_index('feature')
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        top_a = set(a.nsmallest(top_k, 'rank').index)
        top_b = set(b.nsmallest(top_k, 'rank').index)
        union = top_a | top_b
        jaccards.append(len(top_a & top_b) / len(union) if union else 1.0)
        tau = kendalltau(a.loc[common, 'rank'], b.loc[common, 'rank'], nan_policy='omit').statistic
        if np.isfinite(tau):
            taus.append(float(tau))
    return {'pairwise_top_k_jaccard_mean': float(np.mean(jaccards)) if jaccards else np.nan, 'pairwise_top_k_jaccard_min': float(np.min(jaccards)) if jaccards else np.nan, 'pairwise_kendall_tau_mean': float(np.mean(taus)) if taus else np.nan, 'pairwise_kendall_tau_min': float(np.min(taus)) if taus else np.nan, 'n_repeat_pairs': len(jaccards)}

def summarize_oof_shap_stability(fold_table: pd.DataFrame, identity_columns: list[str], all_features: list[str], top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_table.empty:
        return (pd.DataFrame(), pd.DataFrame())
    repeat_rows: list[dict] = []
    identity = {column: fold_table.iloc[0][column] for column in identity_columns}
    for repeat, repeat_folds in fold_table.groupby('repeat', sort=True):
        for feature in all_features:
            feature_rows = repeat_folds[repeat_folds['feature'].eq(feature)]
            weights = feature_rows['n_eval'].to_numpy(dtype=float)
            selected = feature_rows['selected_in_fold'].to_numpy(dtype=bool)
            mean_abs = float(np.average(feature_rows['mean_abs_shap'], weights=weights))
            selected_frequency = float(np.average(selected.astype(float), weights=weights))
            direction_rows = feature_rows[selected & feature_rows['effect_direction'].notna()]
            if direction_rows.empty:
                effect_direction = np.nan
            else:
                effect_direction = float(np.average(direction_rows['effect_direction'], weights=direction_rows['n_eval']))
            repeat_rows.append({**identity, 'repeat': int(repeat), 'feature': feature, 'oof_mean_abs_shap': mean_abs, 'effect_direction': effect_direction, 'selected_frequency_within_repeat': selected_frequency, 'n_oof_samples': int(weights.sum())})
    repeat_table = pd.DataFrame(repeat_rows)
    repeat_table['rank'] = repeat_table.groupby('repeat')['oof_mean_abs_shap'].rank(method='first', ascending=False)
    repeat_table['in_top_k'] = repeat_table['rank'].le(top_k)
    agreement = _pairwise_repeat_agreement(repeat_table, top_k)
    rows = []
    n_features = max(len(all_features), 1)
    for feature, group in repeat_table.groupby('feature', sort=False):
        directions = group['effect_direction'].dropna().to_numpy(dtype=float)
        nonzero_signs = np.sign(directions[np.abs(directions) > 1e-12])
        direction_consistency = float(max(np.mean(nonzero_signs > 0), np.mean(nonzero_signs < 0))) if len(nonzero_signs) else np.nan
        rank_std = float(group['rank'].std(ddof=1)) if len(group) > 1 else 0.0
        rank_stability = max(0.0, 1.0 - rank_std / n_features)
        top_frequency = float(group['in_top_k'].mean())
        selection_frequency = float(group['selected_frequency_within_repeat'].mean())
        direction_for_score = direction_consistency if np.isfinite(direction_consistency) else 0.5
        stability_score = 0.45 * top_frequency + 0.25 * rank_stability + 0.2 * direction_for_score + 0.1 * selection_frequency
        rows.append({**identity, 'feature': feature, 'mean_abs_shap_mean': float(group['oof_mean_abs_shap'].mean()), 'mean_abs_shap_std': float(group['oof_mean_abs_shap'].std(ddof=1)) if len(group) > 1 else 0.0, 'mean_abs_shap_ci_low': float(group['oof_mean_abs_shap'].quantile(0.025)), 'mean_abs_shap_ci_high': float(group['oof_mean_abs_shap'].quantile(0.975)), 'rank_mean': float(group['rank'].mean()), 'rank_std': rank_std, 'rank_ci_low': float(group['rank'].quantile(0.025)), 'rank_ci_high': float(group['rank'].quantile(0.975)), 'top_k_frequency': top_frequency, 'feature_selection_frequency': selection_frequency, 'effect_direction_mean': float(np.mean(directions)) if len(directions) else np.nan, 'effect_direction_consistency': direction_consistency, 'feature_stability_score': stability_score, 'stable_top_feature': bool(top_frequency >= 0.7 and rank_std <= 2.0 and (not np.isfinite(direction_consistency) or direction_consistency >= 0.7)), 'n_repeats': int(group['repeat'].nunique()), **agreement})
    summary = pd.DataFrame(rows).sort_values(['rank_mean', 'feature_stability_score'], ascending=[True, False])
    jaccard = agreement['pairwise_top_k_jaccard_mean']
    tau = agreement['pairwise_kendall_tau_mean']
    if np.isfinite(jaccard) and np.isfinite(tau) and (jaccard >= 0.6) and (tau >= 0.6):
        grade = 'Strong'
    elif np.isfinite(jaccard) and np.isfinite(tau) and (jaccard >= 0.4) and (tau >= 0.4):
        grade = 'Moderate'
    else:
        grade = 'Weak'
    summary['overall_shap_stability_grade'] = grade
    repeat_table['overall_shap_stability_grade'] = grade
    return (summary, repeat_table)
