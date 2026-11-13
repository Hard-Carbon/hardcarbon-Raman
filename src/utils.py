from __future__ import annotations
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)

def deep_update(base: dict, updates: dict) -> dict:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out

def ensure_project_dirs(root: Path, config: dict) -> Path:
    output_dir = resolve_path(root, config.get('output_dir', 'results'))
    subdirs = ['metrics', 'predictions', 'residuals', 'models', 'vif', 'shap', 'robustness', 'low_r2_improvement', 'figures']
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    resolve_path(root, config.get('log_dir', 'logs')).mkdir(parents=True, exist_ok=True)
    return output_dir

def setup_logger(root: Path, config: dict) -> logging.Logger:
    log_dir = resolve_path(root, config.get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('AI4S_target_prediction')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_dir / 'run_log.txt', mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()

def sanitize_name(name: Any) -> str:
    text = str(name)
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
        text = text.replace(char, '_')
    return text

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported table format: {path}')

def save_table(df: pd.DataFrame | None, path: Path, index: bool=False) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding='utf-8-sig')

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def copy_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None

def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
