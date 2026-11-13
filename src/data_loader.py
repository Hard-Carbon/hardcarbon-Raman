from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .utils import read_table, resolve_path, sanitize_name

@dataclass
class TargetDataset:
    target_name: str
    source_path: Path
    data: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]
    target_column: str
    group_values: pd.Series | None
    sample_ids: pd.Series | None
    excluded_target_rows: pd.DataFrame

def _resolve_normalized_files(root: Path, value) -> list[Path]:
    if isinstance(value, list):
        files: list[Path] = []
        for item in value:
            files.extend(_resolve_normalized_files(root, item))
        return sorted(set(files))
    path_text = str(value)
    if any((ch in path_text for ch in ['*', '?', '['])):
        base = root
        return sorted(base.glob(path_text))
    path = resolve_path(root, path_text)
    if path.is_dir():
        files = []
        for pattern in ('*.csv', '*.xlsx', '*.xls'):
            files.extend(path.glob(pattern))
        return sorted(files)
    if path.is_file():
        return [path]
    raise FileNotFoundError(f'normalized_data_path not found: {path}')

def _target_columns_for_file(df: pd.DataFrame, configured) -> list[str]:
    if configured in (None, '', [], '__last__'):
        return [str(df.columns[-1])]
    if isinstance(configured, str):
        return [configured]
    return list(configured)

def _feature_columns_for_target(df: pd.DataFrame, target_col: str, configured_features: list[str] | None, group_col: str | None, sample_id_col: str | None, logger) -> list[str]:
    excluded = {target_col}
    if group_col:
        excluded.add(group_col)
    if sample_id_col:
        excluded.add(sample_id_col)
    if configured_features:
        missing = [col for col in configured_features if col not in df.columns]
        if missing:
            logger.warning('Configured features missing for target %s and will be ignored: %s', target_col, missing)
        candidates = [col for col in configured_features if col in df.columns and col not in excluded]
    else:
        candidates = [col for col in df.columns if col not in excluded]
    numeric_features = []
    ignored = []
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            ignored.append(col)
    if ignored:
        logger.warning('Non-numeric columns ignored for target %s because they are not model features: %s', target_col, ignored)
    return numeric_features

def _validate_no_missing(df: pd.DataFrame, cols: list[str], target_col: str) -> None:
    used_cols = cols + [target_col]
    missing = df[used_cols].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f'Missing values detected in normalized main data. No imputation is performed. Missing counts: {missing.to_dict()}')

def load_normalized_datasets(root: Path, config: dict, logger) -> dict[str, TargetDataset]:
    files = _resolve_normalized_files(root, config['normalized_data_path'])
    if not files:
        raise FileNotFoundError('No normalized CSV/Excel files found.')
    configured_features = config.get('feature_columns') or []
    group_col = config.get('group_column')
    sample_id_col = config.get('sample_id_column')
    datasets: dict[str, TargetDataset] = {}
    for path in files:
        df = read_table(path)
        df.columns = [str(col).strip() for col in df.columns]
        targets = _target_columns_for_file(df, config.get('target_columns'))
        for target_col in targets:
            if target_col not in df.columns:
                raise ValueError(f'Target column {target_col!r} not found in {path}')
            numeric_target = pd.to_numeric(df[target_col], errors='coerce')
            invalid_target = numeric_target.isna()
            excluded_rows = pd.DataFrame()
            if invalid_target.any():
                invalid_values = df.loc[invalid_target, target_col]
                if not bool(config.get('drop_rows_with_invalid_target', False)):
                    raise ValueError(f'Target column contains missing or non-numeric values: {target_col} in {path}. Invalid rows (zero-based): {invalid_values.index.tolist()}; values: {invalid_values.astype(str).tolist()}')
                excluded_rows = pd.DataFrame({'source_file': str(path), 'target_column': target_col, 'original_row_index': invalid_values.index.astype(int), 'excel_row': invalid_values.index.astype(int) + 2, 'original_target_value': invalid_values.astype(str).to_numpy(), 'reason': ['missing_target' if pd.isna(value) else 'non_numeric_target' for value in invalid_values]})
                logger.warning('Excluded %d rows without a numeric supervised target for %s in %s: %s', int(invalid_target.sum()), target_col, path.name, invalid_values.index.tolist())
            target_df = df.loc[~invalid_target].copy()
            target_df[target_col] = numeric_target.loc[~invalid_target].astype(float)
            target_df.reset_index(drop=True, inplace=True)
            features = _feature_columns_for_target(target_df, target_col, configured_features, group_col, sample_id_col, logger)
            if not features:
                raise ValueError(f'No numeric feature columns found for target {target_col} in {path}')
            _validate_no_missing(target_df, features, target_col)
            groups = target_df[group_col] if group_col and group_col in target_df.columns else None
            sample_ids = target_df[sample_id_col] if sample_id_col and sample_id_col in target_df.columns else pd.Series([f'{sanitize_name(path.stem)}_{i}' for i in range(len(target_df))], index=target_df.index, name='sample_id')
            use_file_stem = bool(config.get('target_name_from_file_stem', False))
            target_name = sanitize_name(path.stem) if use_file_stem and len(targets) == 1 else sanitize_name(target_col)
            if use_file_stem and len(targets) > 1:
                target_name = sanitize_name(f'{path.stem}_{target_col}')
            if target_name in datasets:
                target_name = sanitize_name(f'{path.stem}_{target_col}')
            X = target_df[features].copy()
            y = target_df[target_col].copy()
            datasets[target_name] = TargetDataset(target_name=target_name, source_path=path, data=target_df.copy(), X=X, y=y, feature_columns=features, target_column=target_col, group_values=groups, sample_ids=sample_ids, excluded_target_rows=excluded_rows)
            logger.info('Loaded target=%s from %s with n=%d and %d normalized features.', target_name, path.name, len(target_df), len(features))
    return datasets

def collect_excluded_target_rows(datasets: dict[str, TargetDataset]) -> pd.DataFrame:
    frames = []
    for target_name, dataset in datasets.items():
        if dataset.excluded_target_rows.empty:
            continue
        table = dataset.excluded_target_rows.copy()
        table.insert(0, 'target', target_name)
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=['target', 'source_file', 'target_column', 'original_row_index', 'excel_row', 'original_target_value', 'reason'])
    return pd.concat(frames, ignore_index=True)

def summarize_raw_data(root: Path, config: dict, logger) -> pd.DataFrame | None:
    raw_dir_value = config.get('raw_data_dir')
    if not raw_dir_value:
        return None
    raw_dir = resolve_path(root, raw_dir_value)
    rows = []
    if not raw_dir.exists():
        logger.info('Raw data directory does not exist; skipping raw summary: %s', raw_dir)
        return pd.DataFrame([{'raw_data_dir': str(raw_dir), 'status': 'not_found', 'traceability_only': True}])
    files = []
    for pattern in ('*.csv', '*.xlsx', '*.xls'):
        files.extend(raw_dir.glob(pattern))
    if not files:
        return pd.DataFrame([{'raw_data_dir': str(raw_dir), 'status': 'empty', 'traceability_only': True}])
    for path in sorted(files):
        try:
            df = read_table(path)
            rows.append({'file': str(path), 'n_rows': len(df), 'n_columns': df.shape[1], 'columns': ';'.join(map(str, df.columns)), 'missing_values': int(df.isna().sum().sum()), 'traceability_only': True})
        except Exception as exc:
            rows.append({'file': str(path), 'status': 'read_failed', 'error': str(exc), 'traceability_only': True})
    return pd.DataFrame(rows)
