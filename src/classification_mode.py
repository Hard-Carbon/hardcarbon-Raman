from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
from .classification_model_zoo import ClassificationModelSpec
from .model_selection import _rank_score
from .shap_stability import feature_effect_direction, summarize_oof_shap_stability
from .utils import resolve_path, sanitize_name, save_table
from .vif_filter import iterative_vif_filter
from .visualization import save_figure, set_plot_style

@dataclass
class ClassificationTaskDataset:
    target_name: str
    task_id: str
    task_name: str
    source_path: Path
    X: pd.DataFrame
    y_continuous: pd.Series
    y_class: pd.Series
    label_names: dict[int, str]
    sample_ids: pd.Series
    group_values: pd.Series | None
    thresholds: dict[str, float]
    positive_label: int | None

@dataclass
class ClassificationResults:
    labels: pd.DataFrame
    metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion_matrices: pd.DataFrame
    roc_curves: pd.DataFrame
    pr_curves: pd.DataFrame
    best_summary: pd.DataFrame
    model_selection_scores: pd.DataFrame
    unified_selection: pd.DataFrame
    unified_performance: pd.DataFrame
    reliability: pd.DataFrame
    shap_values: pd.DataFrame
    shap_feature_ranking: pd.DataFrame
    shap_stability: pd.DataFrame
    shap_stability_by_repeat: pd.DataFrame

def _cfg(config: dict) -> dict:
    return config.get('classification_settings', {})

def _mean_std_text(mean_value, std_value) -> str:
    if pd.isna(mean_value):
        return ''
    if pd.isna(std_value):
        std_value = 0.0
    return f'{mean_value:.4g} +/- {std_value:.4g}'

def build_classification_tasks(datasets, config: dict, logger) -> tuple[dict[tuple[str, str], ClassificationTaskDataset], pd.DataFrame]:
    cfg = _cfg(config)
    enabled_tasks = set(cfg.get('enabled_tasks', ['task_A_high_vs_others', 'task_B_high_vs_low', 'task_C_low_medium_high']))
    top_q = float(cfg.get('task_a_top_quantile', 0.75))
    high_q = float(cfg.get('task_b_high_quantile', 0.75))
    low_q = float(cfg.get('task_b_low_quantile', 0.25))
    low_med_q = float(cfg.get('task_c_low_quantile', 0.25))
    high_med_q = float(cfg.get('task_c_high_quantile', 0.75))
    tasks: dict[tuple[str, str], ClassificationTaskDataset] = {}
    label_rows: list[dict] = []
    for target_name, dataset in datasets.items():
        y = dataset.y.astype(float).reset_index(drop=True)
        X = dataset.X.reset_index(drop=True)
        sample_ids = dataset.sample_ids.reset_index(drop=True) if dataset.sample_ids is not None else pd.Series([f'{target_name}_{i}' for i in range(len(y))])
        groups = dataset.group_values.reset_index(drop=True) if dataset.group_values is not None else None
        task_defs = []
        high_thr_a = float(y.quantile(top_q))
        label_a = (y >= high_thr_a).astype(int)
        task_defs.append({'task_id': 'task_A_high_vs_others', 'task_name': 'High vs Others', 'labels': label_a, 'included': pd.Series(True, index=y.index), 'label_names': {0: 'non_high', 1: 'high'}, 'thresholds': {'top_quantile': top_q, 'high_threshold': high_thr_a}, 'positive_label': 1})
        high_thr_b = float(y.quantile(high_q))
        low_thr_b = float(y.quantile(low_q))
        included_b = (y >= high_thr_b) | (y <= low_thr_b)
        label_b = pd.Series(np.where(y >= high_thr_b, 1, np.where(y <= low_thr_b, 0, -1)), index=y.index)
        task_defs.append({'task_id': 'task_B_high_vs_low', 'task_name': 'High vs Low', 'labels': label_b, 'included': included_b, 'label_names': {0: 'low', 1: 'high'}, 'thresholds': {'high_quantile': high_q, 'low_quantile': low_q, 'high_threshold': high_thr_b, 'low_threshold': low_thr_b}, 'positive_label': 1})
        low_thr_c = float(y.quantile(low_med_q))
        high_thr_c = float(y.quantile(high_med_q))
        label_c = pd.Series(np.where(y <= low_thr_c, 0, np.where(y >= high_thr_c, 2, 1)), index=y.index)
        task_defs.append({'task_id': 'task_C_low_medium_high', 'task_name': 'Low / Medium / High', 'labels': label_c, 'included': pd.Series(True, index=y.index), 'label_names': {0: 'low', 1: 'medium', 2: 'high'}, 'thresholds': {'low_quantile': low_med_q, 'high_quantile': high_med_q, 'low_threshold': low_thr_c, 'high_threshold': high_thr_c}, 'positive_label': 2})
        for spec in task_defs:
            if spec['task_id'] not in enabled_tasks:
                continue
            labels = spec['labels']
            included = spec['included'].astype(bool)
            for idx in range(len(y)):
                label_value = int(labels.iloc[idx]) if bool(included.iloc[idx]) else -1
                label_rows.append({'target': target_name, 'task_id': spec['task_id'], 'task_name': spec['task_name'], 'sample_index': idx, 'sample_id': sample_ids.iloc[idx], 'source_file': str(dataset.source_path), 'normalized_target_value': float(y.iloc[idx]), 'label': label_value, 'label_name': spec['label_names'].get(label_value, 'excluded_middle'), 'included_in_task': bool(included.iloc[idx]), **spec['thresholds']})
            idx = included[included].index.to_numpy()
            task_y = labels.iloc[idx].astype(int).reset_index(drop=True)
            task = ClassificationTaskDataset(target_name=target_name, task_id=spec['task_id'], task_name=spec['task_name'], source_path=dataset.source_path, X=X.iloc[idx].reset_index(drop=True), y_continuous=y.iloc[idx].reset_index(drop=True), y_class=task_y, label_names=spec['label_names'], sample_ids=sample_ids.iloc[idx].reset_index(drop=True), group_values=groups.iloc[idx].reset_index(drop=True) if groups is not None else None, thresholds=spec['thresholds'], positive_label=spec['positive_label'])
            counts = task_y.value_counts().sort_index().to_dict()
            logger.info('Built classification task target=%s task=%s n=%d counts=%s', target_name, spec['task_id'], len(task_y), counts)
            tasks[target_name, spec['task_id']] = task
    return (tasks, pd.DataFrame(label_rows))

def _build_splits(task: ClassificationTaskDataset, config: dict) -> list[dict]:
    cfg = _cfg(config)
    cv_cfg = cfg.get('cv_settings', config.get('cv_settings', {}))
    y = task.y_class.to_numpy()
    class_counts = pd.Series(y).value_counts()
    if class_counts.empty or len(class_counts) < 2:
        return []
    n_splits = min(int(cv_cfg.get('n_splits', 5)), int(class_counts.min()))
    if n_splits < 2:
        return []
    seed = int(cv_cfg.get('random_seed', 42))
    n_repeats = int(cv_cfg.get('n_repeats', 20))
    indices = np.arange(len(y))
    splits: list[dict] = []
    if task.group_values is not None and cfg.get('include_group_cv', False):
        groups = np.asarray(task.group_values)
        method = str(cv_cfg.get('group_cv_method', 'StratifiedGroupKFold'))
        if method == 'LeaveOneGroupOut':
            splitter = LeaveOneGroupOut()
            iterator = splitter.split(indices, y, groups)
        else:
            unique_groups = np.unique(groups)
            group_splits = min(n_splits, len(unique_groups))
            splitter = StratifiedGroupKFold(n_splits=group_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(indices, y, groups)
        for fold, (train_idx, test_idx) in enumerate(iterator):
            splits.append({'cv_type': method, 'repeat': 0, 'fold': fold, 'train_idx': train_idx, 'test_idx': test_idx})
        return splits
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(splitter.split(indices, y)):
        splits.append({'cv_type': 'repeated_stratified_kfold', 'repeat': i // n_splits, 'fold': i % n_splits, 'train_idx': train_idx, 'test_idx': test_idx})
    return splits

def _select_features_for_split(X: pd.DataFrame, train_idx, test_idx, feature_set: str, config: dict):
    if feature_set == 'vif_filtered_feature_set':
        selected, _, _, _ = iterative_vif_filter(X.iloc[train_idx], threshold=float(config.get('vif_threshold', 10.0)), min_features=int(config.get('min_features_after_vif', 1)))
    else:
        selected = list(X.columns)
    return (X.iloc[train_idx][selected], X.iloc[test_idx][selected], selected)

def _score_matrix(estimator, X, classes: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hasattr(estimator, 'predict_proba'):
        try:
            proba = np.asarray(estimator.predict_proba(X), dtype=float)
            if proba.ndim == 1:
                proba = np.vstack([1.0 - proba, proba]).T
            return (proba, proba)
        except Exception:
            pass
    if hasattr(estimator, 'decision_function'):
        try:
            score = np.asarray(estimator.decision_function(X), dtype=float)
            if score.ndim == 1:
                proba = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
                return (np.vstack([1.0 - proba, proba]).T, score)
            exp = np.exp(score - np.max(score, axis=1, keepdims=True))
            return (exp / np.sum(exp, axis=1, keepdims=True), score)
        except Exception:
            pass
    return (None, None)

def _binary_scores(y_true, y_pred, y_score) -> dict:
    out = {'Accuracy': float(accuracy_score(y_true, y_pred)), 'BalancedAccuracy': float(balanced_accuracy_score(y_true, y_pred)), 'Precision': float(precision_score(y_true, y_pred, zero_division=0)), 'Recall': float(recall_score(y_true, y_pred, zero_division=0)), 'F1': float(f1_score(y_true, y_pred, zero_division=0)), 'MCC': float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan}
    if y_score is not None and len(np.unique(y_true)) > 1:
        out['ROC_AUC'] = float(roc_auc_score(y_true, y_score))
        out['PR_AUC'] = float(average_precision_score(y_true, y_score))
    else:
        out['ROC_AUC'] = np.nan
        out['PR_AUC'] = np.nan
    return out

def _multiclass_scores(y_true, y_pred) -> dict:
    return {'Accuracy': float(accuracy_score(y_true, y_pred)), 'BalancedAccuracy': float(balanced_accuracy_score(y_true, y_pred)), 'Precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 'Recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 'F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'MCC': float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan, 'ROC_AUC': np.nan, 'PR_AUC': np.nan}

def _curve_rows(meta: dict, y_true, y_score) -> tuple[list[dict], list[dict]]:
    roc_rows: list[dict] = []
    pr_rows: list[dict] = []
    if y_score is None or len(np.unique(y_true)) < 2:
        return (roc_rows, pr_rows)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        for i, (fp, tp, thr) in enumerate(zip(fpr, tpr, thresholds)):
            roc_rows.append({**meta, 'point_index': i, 'fpr': float(fp), 'tpr': float(tp), 'threshold': float(thr) if np.isfinite(thr) else np.inf})
    except Exception:
        pass
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        padded_thresholds = list(thresholds) + [np.nan]
        for i, (pr, re, thr) in enumerate(zip(precision, recall, padded_thresholds)):
            pr_rows.append({**meta, 'point_index': i, 'precision': float(pr), 'recall': float(re), 'threshold': float(thr) if pd.notna(thr) else np.nan})
    except Exception:
        pass
    return (roc_rows, pr_rows)

def _aggregate_metrics(fold_df: pd.DataFrame, spec: ClassificationModelSpec) -> list[dict]:
    if fold_df.empty:
        return []
    rows = []
    group_cols = ['target', 'task_id', 'task_name', 'model', 'feature_set', 'cv_type']
    for key, group in fold_df.groupby(group_cols, dropna=False):
        success = group[group['fit_status'] == 'success']
        row = dict(zip(group_cols, key))
        row.update({'category': spec.category, 'complexity_rank': spec.complexity_rank, 'explanation_candidate': spec.explanation_candidate, 'n_total_folds': len(group), 'n_success_folds': len(success), 'n_failed_folds': int((group['fit_status'] != 'success').sum()), 'failure_rate': float((group['fit_status'] != 'success').mean()), 'mean_n_features': float(success['n_features'].mean()) if not success.empty and 'n_features' in success else np.nan, 'n_classes': int(success['n_classes'].max()) if not success.empty and 'n_classes' in success else np.nan})
        for metric in ['Accuracy', 'BalancedAccuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'MCC']:
            row[f'{metric}_mean'] = float(success[metric].mean()) if not success.empty and metric in success else np.nan
            row[f'{metric}_std'] = float(success[metric].std(ddof=1)) if len(success) > 1 and metric in success else 0.0
            row[f'{metric}_mean_std'] = _mean_std_text(row[f'{metric}_mean'], row[f'{metric}_std'])
        rows.append(row)
    return rows

def evaluate_classification_tasks(tasks, model_specs: dict[str, ClassificationModelSpec], config: dict, logger):
    feature_sets = config.get('classification_settings', {}).get('feature_sets', config.get('feature_sets', ['raw_feature_set', 'vif_filtered_feature_set']))
    seed0 = int(_cfg(config).get('cv_settings', config.get('cv_settings', {})).get('random_seed', 42))
    metric_rows: list[dict] = []
    fold_rows: list[dict] = []
    pred_rows: list[dict] = []
    confusion_rows: list[dict] = []
    roc_rows: list[dict] = []
    pr_rows: list[dict] = []
    for (target, task_id), task in tasks.items():
        splits = _build_splits(task, config)
        if not splits:
            logger.warning('Skipping classification task with insufficient class counts: target=%s task=%s', target, task_id)
            continue
        classes = np.sort(task.y_class.unique())
        n_classes = len(classes)
        for feature_set in feature_sets:
            feature_cache = {}
            for model_name, spec in model_specs.items():
                logger.info('Classification | target=%s | task=%s | feature_set=%s | model=%s', target, task_id, feature_set, model_name)
                local_fold_rows = []
                for split in splits:
                    seed = seed0 + int(split['repeat']) * 1000 + int(split['fold'])
                    meta = {'target': target, 'task_id': task_id, 'task_name': task.task_name, 'model': model_name, 'feature_set': feature_set, 'cv_type': split['cv_type'], 'repeat': split['repeat'], 'fold': split['fold']}
                    try:
                        cache_key = (int(split['repeat']), int(split['fold']))
                        if cache_key not in feature_cache:
                            feature_cache[cache_key] = _select_features_for_split(task.X, split['train_idx'], split['test_idx'], feature_set, config)
                        X_train, X_test, selected = feature_cache[cache_key]
                        y_train = task.y_class.iloc[split['train_idx']]
                        y_test = task.y_class.iloc[split['test_idx']]
                        estimator = spec.create(seed, X_train.shape[1], X_train.shape[0], n_classes)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            estimator.fit(X_train, y_train)
                            y_pred = np.asarray(estimator.predict(X_test)).ravel().astype(int)
                            proba, raw_score = _score_matrix(estimator, X_test, classes)
                        if n_classes == 2:
                            if proba is not None and proba.shape[1] >= 2:
                                y_score = proba[:, list(classes).index(task.positive_label)] if task.positive_label in classes else proba[:, -1]
                            elif raw_score is not None and np.asarray(raw_score).ndim == 1:
                                y_score = np.asarray(raw_score).ravel()
                            else:
                                y_score = None
                            metrics = _binary_scores(y_test, y_pred, y_score)
                        else:
                            y_score = None
                            metrics = _multiclass_scores(y_test, y_pred)
                        fold_row = {**meta, **metrics, 'fit_status': 'success', 'error': '', 'n_features': len(selected), 'selected_features': ';'.join(selected), 'n_classes': n_classes}
                        local_fold_rows.append(fold_row)
                        fold_rows.append(fold_row)
                        cm = confusion_matrix(y_test, y_pred, labels=classes)
                        for i, true_label in enumerate(classes):
                            for j, pred_label in enumerate(classes):
                                confusion_rows.append({**meta, 'true_label': int(true_label), 'true_label_name': task.label_names.get(int(true_label), str(true_label)), 'predicted_label': int(pred_label), 'predicted_label_name': task.label_names.get(int(pred_label), str(pred_label)), 'count': int(cm[i, j])})
                        curve_meta = {**meta, 'positive_label': task.positive_label, 'positive_label_name': task.label_names.get(int(task.positive_label), '') if task.positive_label is not None else ''}
                        fold_roc, fold_pr = _curve_rows(curve_meta, y_test, y_score) if n_classes == 2 else ([], [])
                        roc_rows.extend(fold_roc)
                        pr_rows.extend(fold_pr)
                        for local_pos, sample_index in enumerate(split['test_idx']):
                            score_value = float(y_score[local_pos]) if y_score is not None else np.nan
                            pred_rows.append({**meta, 'sample_index_task': int(sample_index), 'sample_id': task.sample_ids.iloc[sample_index], 'source_file': str(task.source_path), 'normalized_target_value': float(task.y_continuous.iloc[sample_index]), 'true_label': int(y_test.iloc[local_pos]), 'true_label_name': task.label_names.get(int(y_test.iloc[local_pos]), str(y_test.iloc[local_pos])), 'predicted_label': int(y_pred[local_pos]), 'predicted_label_name': task.label_names.get(int(y_pred[local_pos]), str(y_pred[local_pos])), 'positive_class_score': score_value, 'correct': bool(int(y_test.iloc[local_pos]) == int(y_pred[local_pos]))})
                    except Exception as exc:
                        logger.warning('Classification model failed | target=%s task=%s model=%s feature_set=%s repeat=%s fold=%s error=%s', target, task_id, model_name, feature_set, split['repeat'], split['fold'], exc)
                        fold_row = {**meta, 'Accuracy': np.nan, 'BalancedAccuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan, 'ROC_AUC': np.nan, 'PR_AUC': np.nan, 'MCC': np.nan, 'fit_status': 'failed', 'error': str(exc), 'n_features': np.nan, 'selected_features': '', 'n_classes': n_classes}
                        local_fold_rows.append(fold_row)
                        fold_rows.append(fold_row)
                metric_rows.extend(_aggregate_metrics(pd.DataFrame(local_fold_rows), spec))
    return (pd.DataFrame(metric_rows), pd.DataFrame(fold_rows), pd.DataFrame(pred_rows), pd.DataFrame(confusion_rows), pd.DataFrame(roc_rows), pd.DataFrame(pr_rows))

def _classification_score(group: pd.DataFrame) -> pd.DataFrame:
    scored = group.copy()
    n_classes = int(scored['n_classes'].max()) if 'n_classes' in scored and (not scored['n_classes'].isna().all()) else 2
    scored['BalancedAccuracy_rank_score'] = _rank_score(scored['BalancedAccuracy_mean'], True)
    scored['F1_rank_score'] = _rank_score(scored['F1_mean'], True)
    scored['MCC_rank_score'] = _rank_score(scored['MCC_mean'], True)
    if n_classes == 2:
        scored['ROC_AUC_rank_score'] = _rank_score(scored['ROC_AUC_mean'], True)
        scored['PR_AUC_rank_score'] = _rank_score(scored['PR_AUC_mean'], True)
        scored['classification_score'] = 0.3 * scored['BalancedAccuracy_rank_score'] + 0.25 * scored['F1_rank_score'] + 0.2 * scored['ROC_AUC_rank_score'] + 0.15 * scored['PR_AUC_rank_score'] + 0.1 * scored['MCC_rank_score']
    else:
        scored['ROC_AUC_rank_score'] = np.nan
        scored['PR_AUC_rank_score'] = np.nan
        scored['classification_score'] = 0.4 * scored['BalancedAccuracy_rank_score'] + 0.35 * scored['F1_rank_score'] + 0.25 * scored['MCC_rank_score']
    stability = (_rank_score(scored['BalancedAccuracy_std'], False) + _rank_score(scored['F1_std'], False) + _rank_score(scored['MCC_std'], False)) / 3.0
    scored['stability_rank_score'] = stability
    return scored.sort_values('classification_score', ascending=False)

def select_best_classification_models(metrics: pd.DataFrame, config: dict, logger):
    if metrics.empty:
        return (pd.DataFrame(), pd.DataFrame())
    primary_cv = _cfg(config).get('primary_cv_type', 'repeated_stratified_kfold')
    primary = metrics[(metrics['cv_type'] == primary_cv) & (metrics['n_success_folds'] > 0)].copy()
    scored_frames = []
    summary_rows = []
    for (target, task_id), group in primary.groupby(['target', 'task_id'], dropna=False):
        scored = _classification_score(group)
        scored_frames.append(scored)
        top = scored.iloc[0]
        if _cfg(config).get('one_standard_error', True):
            n = max(float(top.get('n_success_folds', 1)), 1.0)
            ba_se = float(top.get('BalancedAccuracy_std', 0.0)) / math.sqrt(n)
            f1_se = float(top.get('F1_std', 0.0)) / math.sqrt(n)
            mcc_se = float(top.get('MCC_std', 0.0)) / math.sqrt(n)
            within = scored[(scored['BalancedAccuracy_mean'] >= top['BalancedAccuracy_mean'] - ba_se) & (scored['F1_mean'] >= top['F1_mean'] - f1_se) & (scored['MCC_mean'] >= top['MCC_mean'] - mcc_se)].copy()
            if not within.empty:
                within = within.sort_values(['failure_rate', 'stability_rank_score', 'complexity_rank', 'classification_score'], ascending=[True, False, True, False])
                selected = within.iloc[0]
                reason = 'Selected by one-standard-error preference among comparable classifiers.'
            else:
                selected = top
                reason = 'Selected highest classification score.'
        else:
            selected = top
            reason = 'Selected highest classification score.'
        row = {'target': target, 'task_id': task_id, 'task_name': selected.get('task_name'), 'best_model': selected['model'], 'best_feature_set': selected['feature_set'], 'category': selected.get('category'), 'classification_score': selected.get('classification_score'), 'n_classes': selected.get('n_classes'), 'BalancedAccuracy_mean': selected.get('BalancedAccuracy_mean'), 'BalancedAccuracy_std': selected.get('BalancedAccuracy_std'), 'F1_mean': selected.get('F1_mean'), 'F1_std': selected.get('F1_std'), 'ROC_AUC_mean': selected.get('ROC_AUC_mean'), 'ROC_AUC_std': selected.get('ROC_AUC_std'), 'PR_AUC_mean': selected.get('PR_AUC_mean'), 'PR_AUC_std': selected.get('PR_AUC_std'), 'MCC_mean': selected.get('MCC_mean'), 'MCC_std': selected.get('MCC_std'), 'Accuracy_mean': selected.get('Accuracy_mean'), 'Precision_mean': selected.get('Precision_mean'), 'Recall_mean': selected.get('Recall_mean'), 'failure_rate': selected.get('failure_rate'), 'mean_n_features': selected.get('mean_n_features'), 'selection_reason': reason}
        summary_rows.append(row)
        logger.info('Best classifier target=%s task=%s: %s (%s), BA=%.4f F1=%.4f MCC=%.4f', target, task_id, selected['model'], selected['feature_set'], selected.get('BalancedAccuracy_mean', np.nan), selected.get('F1_mean', np.nan), selected.get('MCC_mean', np.nan))
    return (pd.DataFrame(summary_rows), pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame())

def select_unified_classification_model(metrics: pd.DataFrame, best_summary: pd.DataFrame, model_specs, config: dict, logger):
    candidates = [m for m in _cfg(config).get('unified_candidate_models', ['CatBoostClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'XGBClassifier', 'LGBMClassifier']) if m in model_specs]
    if metrics.empty or not candidates:
        return (pd.DataFrame(), pd.DataFrame())
    primary_cv = _cfg(config).get('primary_cv_type', 'repeated_stratified_kfold')
    preferred_feature_set = _cfg(config).get('unified_preferred_feature_set', 'vif_filtered_feature_set')
    data = metrics[(metrics['cv_type'] == primary_cv) & metrics['model'].isin(candidates) & (metrics['n_success_folds'] > 0)].copy()
    if data.empty:
        return (pd.DataFrame(), pd.DataFrame())
    data['_feature_set_priority'] = np.select([data['feature_set'].eq(preferred_feature_set), data['feature_set'].eq('raw_feature_set')], [0, 1], default=2)
    data = data.sort_values('_feature_set_priority').drop_duplicates(['target', 'task_id', 'model'], keep='first').drop(columns='_feature_set_priority')
    rows = []
    for model, group in data.groupby('model'):
        rows.append({'candidate_model': model, 'n_tasks_evaluated': group[['target', 'task_id']].drop_duplicates().shape[0], 'average_BalancedAccuracy_mean': group['BalancedAccuracy_mean'].mean(), 'average_F1_mean': group['F1_mean'].mean(), 'average_MCC_mean': group['MCC_mean'].mean(), 'average_ROC_AUC_mean': group['ROC_AUC_mean'].mean(skipna=True), 'average_PR_AUC_mean': group['PR_AUC_mean'].mean(skipna=True), 'average_stability': group[['BalancedAccuracy_std', 'F1_std', 'MCC_std']].mean(axis=1).mean(), 'mean_failure_rate': group['failure_rate'].mean(), 'feature_set': group['feature_set'].mode().iloc[0]})
    selection = pd.DataFrame(rows)
    selection['BA_rank_score'] = _rank_score(selection['average_BalancedAccuracy_mean'], True)
    selection['F1_rank_score'] = _rank_score(selection['average_F1_mean'], True)
    selection['MCC_rank_score'] = _rank_score(selection['average_MCC_mean'], True)
    selection['stability_rank_score'] = _rank_score(selection['average_stability'], False)
    selection['unified_classification_score'] = 0.35 * selection['BA_rank_score'] + 0.3 * selection['F1_rank_score'] + 0.2 * selection['MCC_rank_score'] + 0.15 * selection['stability_rank_score']
    selection = selection.sort_values('unified_classification_score', ascending=False)
    selected_model = selection.iloc[0]['candidate_model']
    performance = data[data['model'] == selected_model].copy()
    performance['selected_unified_model'] = selected_model
    logger.info('Unified classification explanation model: %s', selected_model)
    return (selection, performance)

def grade_classification_reliability(best_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if best_summary.empty:
        return pd.DataFrame()
    for _, row in best_summary.iterrows():
        n_classes = int(row.get('n_classes', 2))
        ba = float(row.get('BalancedAccuracy_mean', np.nan))
        roc = float(row.get('ROC_AUC_mean', np.nan))
        mcc = float(row.get('MCC_mean', np.nan))
        ba_std = float(row.get('BalancedAccuracy_std', np.nan))
        shap_stable_proxy = bool(pd.notna(ba_std) and ba_std <= 0.15)
        if n_classes == 2:
            if ba >= 0.7 and roc >= 0.75 and (mcc >= 0.4) and shap_stable_proxy:
                grade, usage = ('A', 'main_text_key_result')
            elif ba >= 0.65 and roc >= 0.7 and (mcc >= 0.3):
                grade, usage = ('B', 'main_text_supporting_or_SI')
            elif ba > 0.53 or roc > 0.6 or mcc > 0.1:
                grade, usage = ('C', 'exploratory_result')
            else:
                grade, usage = ('D', 'do_not_make_strong_mechanistic_claim')
        elif ba >= 0.7 and mcc >= 0.4 and shap_stable_proxy:
            grade, usage = ('A', 'main_text_key_result')
        elif ba >= 0.65 and mcc >= 0.3:
            grade, usage = ('B', 'main_text_supporting_or_SI')
        elif ba > 0.42 or mcc > 0.1:
            grade, usage = ('C', 'exploratory_result')
        else:
            grade, usage = ('D', 'do_not_make_strong_mechanistic_claim')
        rows.append({'target': row['target'], 'task_id': row['task_id'], 'task_name': row['task_name'], 'best_model': row['best_model'], 'best_feature_set': row['best_feature_set'], 'grade': grade, 'recommended_usage': usage, 'BalancedAccuracy_mean': ba, 'F1_mean': row.get('F1_mean'), 'ROC_AUC_mean': row.get('ROC_AUC_mean'), 'PR_AUC_mean': row.get('PR_AUC_mean'), 'MCC_mean': mcc, 'BalancedAccuracy_std': ba_std, 'shap_stability_proxy': shap_stable_proxy})
    return pd.DataFrame(rows)

def _final_features_for_task(task: ClassificationTaskDataset, feature_set: str, config: dict):
    if feature_set == 'vif_filtered_feature_set':
        selected, _, _, _ = iterative_vif_filter(task.X, threshold=float(config.get('vif_threshold', 10.0)), min_features=int(config.get('min_features_after_vif', 1)))
        return selected
    return list(task.X.columns)

def _as_class_shap_values(values, class_index: int):
    if isinstance(values, list):
        idx = min(class_index, len(values) - 1)
        return np.asarray(values[idx])
    arr = np.asarray(values)
    if arr.ndim == 3:
        idx = min(class_index, arr.shape[2] - 1)
        return arr[:, :, idx]
    return arr

def run_classification_shap(tasks, model_specs, unified_performance, reliability, root, config, logger):
    if not _cfg(config).get('shap_enabled', True):
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    try:
        import matplotlib.pyplot as plt
        import shap
    except Exception as exc:
        logger.warning('SHAP not available for classification workflow: %s', exc)
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if unified_performance is None or unified_performance.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    set_plot_style(config)
    output_dir = resolve_path(root, config.get('output_dir', 'results')) / 'classification' / 'shap'
    seed = int(_cfg(config).get('cv_settings', config.get('cv_settings', {})).get('random_seed', 42))
    max_samples = int(_cfg(config).get('shap_max_samples', 100))
    max_display = int(_cfg(config).get('shap_max_display', 15))
    dependence_top_n = int(_cfg(config).get('shap_dependence_top_n', 6))
    reliability_lookup = reliability.set_index(['target', 'task_id']) if reliability is not None and (not reliability.empty) else None
    shap_frames = []
    ranking_frames = []
    stability_frames = []
    stability_repeat_frames = []
    for _, perf in unified_performance.drop_duplicates(['target', 'task_id']).iterrows():
        key = (perf['target'], perf['task_id'])
        if key not in tasks:
            continue
        task = tasks[key]
        grade = 'D'
        if reliability_lookup is not None and key in reliability_lookup.index:
            grade = reliability_lookup.loc[key].get('grade', 'D')
        scope = 'supporting_information' if grade == 'D' else 'main'
        model_name = perf['selected_unified_model']
        feature_set = perf['feature_set']
        if model_name not in model_specs:
            continue
        try:
            selected = _final_features_for_task(task, feature_set, config)
            X = task.X[selected]
            y = task.y_class
            estimator = model_specs[model_name].create(seed, X.shape[1], X.shape[0], len(np.unique(y)))
            estimator.fit(X, y)
            if len(X) > max_samples:
                X_eval = X.sample(n=max_samples, random_state=seed).sort_index()
            else:
                X_eval = X.copy()
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(X_eval)
            high_class = int(task.positive_label if task.positive_label is not None else max(task.label_names))
            class_index = list(np.sort(task.y_class.unique())).index(high_class) if high_class in set(task.y_class.unique()) else len(np.unique(y)) - 1
            shap_values = _as_class_shap_values(values, class_index)
            expected = explainer.expected_value
            if isinstance(expected, (list, tuple, np.ndarray)):
                expected_value = float(np.asarray(expected).ravel()[min(class_index, len(np.asarray(expected).ravel()) - 1)])
            else:
                expected_value = float(expected)
            target_dir = output_dir / sanitize_name(task.target_name) / sanitize_name(task.task_id) / scope
            target_dir.mkdir(parents=True, exist_ok=True)
            value_cols = {f'value_{col}': X_eval[col].to_numpy() for col in X_eval.columns}
            shap_cols = {f'shap_{col}': shap_values[:, i] for i, col in enumerate(X_eval.columns)}
            shap_df = pd.DataFrame({'target': task.target_name, 'task_id': task.task_id, 'task_name': task.task_name, 'scope': scope, 'grade': grade, 'model': model_name, 'feature_set': feature_set, 'explained_class': high_class, 'explained_class_name': task.label_names.get(high_class, str(high_class)), 'sample_index_task': X_eval.index.to_numpy(), 'expected_value': expected_value, **value_cols, **shap_cols})
            mean_abs = np.abs(shap_values).mean(axis=0)
            ranking = pd.DataFrame({'target': task.target_name, 'task_id': task.task_id, 'task_name': task.task_name, 'scope': scope, 'grade': grade, 'model': model_name, 'feature_set': feature_set, 'feature': list(X_eval.columns), 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
            ranking['rank'] = np.arange(1, len(ranking) + 1)
            shap_frames.append(shap_df)
            ranking_frames.append(ranking)
            save_table(shap_df, target_dir / 'classification_shap_values.csv')
            save_table(ranking, target_dir / 'classification_shap_feature_ranking.csv')
            stability_cfg = _cfg(config).get('shap_stability', {})
            if stability_cfg.get('enabled', True):
                fold_rows = []
                stability_splits = [split for split in _build_splits(task, config) if split['cv_type'] == 'repeated_stratified_kfold']
                max_folds = stability_cfg.get('max_folds')
                if max_folds is not None:
                    stability_splits = stability_splits[:int(max_folds)]
                top_k = int(stability_cfg.get('top_k', 5))
                for split in stability_splits:
                    try:
                        X_train, X_test, fold_features = _select_features_for_split(task.X, split['train_idx'], split['test_idx'], feature_set, config)
                        y_train = task.y_class.iloc[split['train_idx']]
                        fold_seed = seed + 1000 * int(split['repeat']) + int(split['fold'])
                        fold_model = model_specs[model_name].create(fold_seed, X_train.shape[1], X_train.shape[0], len(np.unique(y_train)))
                        fold_model.fit(X_train, y_train)
                        fold_explainer = shap.TreeExplainer(fold_model)
                        fold_values = _as_class_shap_values(fold_explainer.shap_values(X_test), class_index)
                        fold_mean_abs = dict(zip(fold_features, np.abs(fold_values).mean(axis=0)))
                        fold_directions = {feature: feature_effect_direction(X_test[feature].to_numpy(), fold_values[:, i]) for i, feature in enumerate(fold_features)}
                        for feature in task.X.columns:
                            fold_rows.append({'target': task.target_name, 'task_id': task.task_id, 'task_name': task.task_name, 'scope': scope, 'grade': grade, 'model': model_name, 'feature_set': feature_set, 'cv_type': split['cv_type'], 'repeat': split['repeat'], 'fold': split['fold'], 'feature': feature, 'mean_abs_shap': fold_mean_abs.get(feature, 0.0), 'effect_direction': fold_directions.get(feature, np.nan), 'selected_in_fold': feature in fold_features, 'n_eval': len(X_test)})
                    except Exception as exc:
                        logger.warning('Classification SHAP stability fold failed for %s %s repeat=%s fold=%s: %s', task.target_name, task.task_id, split['repeat'], split['fold'], exc)
                fold_table = pd.DataFrame(fold_rows)
                stability, stability_repeats = summarize_oof_shap_stability(fold_table, ['target', 'task_id', 'task_name', 'scope', 'grade', 'model', 'feature_set', 'cv_type'], list(task.X.columns), top_k)
                if not stability.empty:
                    stability_frames.append(stability)
                    stability_repeat_frames.append(stability_repeats)
                    save_table(stability, target_dir / 'classification_shap_stability_across_folds.csv')
                    save_table(stability_repeats, target_dir / 'classification_shap_stability_by_repeat.csv')
                    top_stability = stability.nsmallest(max_display, 'rank_mean').sort_values('rank_mean', ascending=False)
                    fig, ax = plt.subplots(figsize=(config.get('plotting_settings', {}).get('wide_figure_width', 7.0), config.get('plotting_settings', {}).get('wide_figure_height', 4.2)))
                    colors = plt.cm.viridis(np.clip(top_stability['effect_direction_consistency'].fillna(0.5), 0, 1))
                    ax.barh(top_stability['feature'], top_stability['top_k_frequency'], color=colors)
                    ax.axvline(0.7, color='black', linestyle='--', linewidth=0.9)
                    ax.set_xlim(0, 1.02)
                    ax.set_xlabel('frequency in SHAP top-k across stratified CV repeats')
                    ax.set_ylabel('normalized descriptor')
                    ax.set_title(f'{task.target_name} | {task.task_name} | OOF SHAP stability')
                    save_figure(fig, target_dir / 'classification_shap_cross_fold_stability', config)
                    plt.close(fig)
            plot_cfg = config.get('plotting_settings', {})
            top = ranking.head(max_display).iloc[::-1]
            fig, ax = plt.subplots(figsize=(plot_cfg.get('figure_width', 3.5), plot_cfg.get('figure_height', 3.0)))
            ax.barh(top['feature'], top['mean_abs_shap'], color='#4C78A8')
            ax.set_xlabel('mean |SHAP value| for high-performance class')
            ax.set_ylabel('normalized descriptor')
            ax.set_title(f'{task.target_name} | {task.task_name}')
            save_figure(fig, target_dir / 'classification_shap_bar_plot', config)
            plt.close(fig)
            try:
                shap.summary_plot(shap_values, X_eval, max_display=max_display, show=False)
                fig = plt.gcf()
                plt.gca().set_xlabel('SHAP value for high-performance class')
                save_figure(fig, target_dir / 'classification_shap_beeswarm_plot', config)
                plt.close(fig)
            except Exception as exc:
                logger.warning('Classification SHAP beeswarm failed for %s %s: %s', task.target_name, task.task_id, exc)
            for feature in ranking.head(min(dependence_top_n, len(ranking)))['feature']:
                feature_idx = list(X_eval.columns).index(feature)
                color_feature = ranking.iloc[1]['feature'] if len(ranking) > 1 and ranking.iloc[0]['feature'] == feature else ranking.iloc[0]['feature']
                fig, ax = plt.subplots(figsize=(plot_cfg.get('figure_width', 3.5), plot_cfg.get('figure_height', 3.0)))
                sc = ax.scatter(X_eval[feature], shap_values[:, feature_idx], c=X_eval[color_feature], cmap='viridis', s=float(plot_cfg.get('marker_size', 28)), edgecolor='white', linewidth=0.3)
                ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)
                ax.set_xlabel(f'{feature} (normalized descriptor value)')
                ax.set_ylabel('SHAP value for high-performance class')
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(f'{color_feature} normalized descriptor value')
                save_figure(fig, target_dir / f'classification_shap_dependence_{sanitize_name(feature)}', config)
                plt.close(fig)
            try:
                sample_idx = 0
                explanation = shap.Explanation(values=shap_values[sample_idx], base_values=expected_value, data=X_eval.iloc[sample_idx].to_numpy(), feature_names=list(X_eval.columns))
                shap.plots.waterfall(explanation, max_display=max_display, show=False)
                fig = plt.gcf()
                plt.gca().set_xlabel('SHAP contribution to high-performance class')
                save_figure(fig, target_dir / 'classification_shap_waterfall_plot', config)
                plt.close(fig)
            except Exception as exc:
                logger.warning('Classification SHAP waterfall failed for %s %s: %s', task.target_name, task.task_id, exc)
        except Exception as exc:
            logger.warning('Classification SHAP failed for target=%s task=%s model=%s: %s', task.target_name, task.task_id, model_name, exc)
    return (pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame(), pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame(), pd.concat(stability_frames, ignore_index=True) if stability_frames else pd.DataFrame(), pd.concat(stability_repeat_frames, ignore_index=True) if stability_repeat_frames else pd.DataFrame())

def apply_measured_shap_stability(reliability: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    if reliability is None or reliability.empty or stability is None or stability.empty:
        return reliability
    task_stability = stability.groupby(['target', 'task_id'], as_index=False).agg(shap_stability_grade=('overall_shap_stability_grade', 'first'), shap_top_k_jaccard=('pairwise_top_k_jaccard_mean', 'first'), shap_kendall_tau=('pairwise_kendall_tau_mean', 'first'), n_stable_top_features=('stable_top_feature', 'sum'))
    output = reliability.drop(columns=[column for column in task_stability.columns if column in reliability.columns and column not in {'target', 'task_id'}], errors='ignore').merge(task_stability, on=['target', 'task_id'], how='left')
    output['shap_stability_proxy'] = output['shap_stability_grade'].isin(['Strong', 'Moderate'])
    grade_a_invalid = output['grade'].eq('A') & ~output['shap_stability_proxy']
    output.loc[grade_a_invalid, 'grade'] = 'B'
    output.loc[grade_a_invalid, 'recommended_usage'] = 'main_text_supporting_or_SI'
    output['shap_interpretation_scope'] = np.select([output['shap_stability_grade'].eq('Strong'), output['shap_stability_grade'].eq('Moderate')], ['main_interpretation', 'supporting_interpretation'], default='exploratory_only')
    return output

def run_classification_workflow(datasets, model_specs, root, config, logger) -> ClassificationResults:
    tasks, labels = build_classification_tasks(datasets, config, logger)
    metrics, fold_metrics, predictions, confusion, roc_curves, pr_curves = evaluate_classification_tasks(tasks, model_specs, config, logger)
    best_summary, scored = select_best_classification_models(metrics, config, logger)
    unified_selection, unified_performance = select_unified_classification_model(metrics, best_summary, model_specs, config, logger)
    reliability = grade_classification_reliability(best_summary)
    shap_values, shap_ranking, shap_stability, shap_stability_by_repeat = run_classification_shap(tasks, model_specs, unified_performance, reliability, root, config, logger)
    reliability = apply_measured_shap_stability(reliability, shap_stability)
    return ClassificationResults(labels=labels, metrics=metrics, fold_metrics=fold_metrics, predictions=predictions, confusion_matrices=confusion, roc_curves=roc_curves, pr_curves=pr_curves, best_summary=best_summary, model_selection_scores=scored, unified_selection=unified_selection, unified_performance=unified_performance, reliability=reliability, shap_values=shap_values, shap_feature_ranking=shap_ranking, shap_stability=shap_stability, shap_stability_by_repeat=shap_stability_by_repeat)
