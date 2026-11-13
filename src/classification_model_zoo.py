from __future__ import annotations
import importlib.util
import os
import tempfile
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

@dataclass
class ClassificationModelSpec:
    name: str
    category: str
    complexity_rank: int
    explanation_candidate: bool
    create: Callable[[int, int, int, int], object]
    notes: str = ''

def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def _enabled(name: str, enabled_models) -> bool:
    return enabled_models == 'all' or name in set(enabled_models)

def _default(config: dict, key: str, fallback):
    return config.get('classification_settings', {}).get('defaults', {}).get(key, config.get('model_settings', {}).get('defaults', {}).get(key, fallback))

def _n_jobs(config: dict) -> int:
    return int(config.get('classification_settings', {}).get('n_jobs', config.get('model_settings', {}).get('n_jobs', 1)))

def _scaled(estimator):
    return Pipeline([('scaler', StandardScaler()), ('model', estimator)])

def _bagging_classifier(random_state: int, n_estimators: int, n_jobs: int):
    kwargs = {'n_estimators': n_estimators, 'random_state': random_state, 'n_jobs': n_jobs}
    try:
        return BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), **kwargs)

def _xgb_classifier(random_state: int, n_estimators: int, max_depth: int, learning_rate: float, n_jobs: int, n_classes: int):
    kwargs = {'n_estimators': n_estimators, 'max_depth': min(max_depth, 4), 'learning_rate': learning_rate, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_lambda': 1.0, 'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic', 'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss', 'random_state': random_state, 'n_jobs': n_jobs, 'verbosity': 0}
    if n_classes > 2:
        kwargs['num_class'] = n_classes
    return __import__('xgboost').XGBClassifier(**kwargs)

def build_classification_model_zoo(config: dict, logger) -> tuple[dict[str, ClassificationModelSpec], list[dict]]:
    cfg = config.get('classification_settings', {})
    enabled_models = cfg.get('enabled_models', 'all')
    allow_automl = bool(cfg.get('allow_automl_models', False))
    n_estimators = int(_default(config, 'n_estimators', 200))
    max_depth = int(_default(config, 'tree_max_depth', 4))
    learning_rate = float(_default(config, 'learning_rate', 0.03))
    max_iter = int(_default(config, 'max_iter', 1000))
    n_jobs = _n_jobs(config)
    specs: dict[str, ClassificationModelSpec] = {}
    unavailable: list[dict] = []

    def add(name, category, complexity, explanation_candidate, factory, package=None, reason=None, require_token_env=None):
        if not _enabled(name, enabled_models):
            return
        if package and (not _has_module(package)):
            unavailable.append({'model': name, 'package': package, 'reason': reason or f'Missing optional package: {package}'})
            logger.warning('Skipping %s because optional package %s is not installed.', name, package)
            return
        if require_token_env and (not os.environ.get(require_token_env)):
            unavailable.append({'model': name, 'package': package or '', 'reason': f'Missing {require_token_env}; skipping to avoid license/auth failure.'})
            logger.warning('Skipping %s because %s is not set.', name, require_token_env)
            return
        specs[name] = ClassificationModelSpec(name, category, complexity, explanation_candidate, factory)
    add('LogisticRegression', 'linear', 2, False, lambda rs, nf, ns, nc: _scaled(LogisticRegression(max_iter=max_iter, class_weight='balanced', random_state=rs, n_jobs=None)))
    add('RidgeClassifier', 'linear', 2, False, lambda rs, nf, ns, nc: _scaled(RidgeClassifier(class_weight='balanced')))
    add('LinearSVC', 'linear', 3, False, lambda rs, nf, ns, nc: _scaled(LinearSVC(class_weight='balanced', random_state=rs, max_iter=max_iter)))
    add('SVC_linear', 'kernel_distance', 4, False, lambda rs, nf, ns, nc: _scaled(SVC(kernel='linear', C=float(_default(config, 'svr_c', 10.0)), class_weight='balanced', probability=False, random_state=rs)))
    add('SVC_rbf', 'kernel_distance', 5, False, lambda rs, nf, ns, nc: _scaled(SVC(kernel='rbf', C=float(_default(config, 'svr_c', 10.0)), gamma=_default(config, 'svr_gamma', 'scale'), class_weight='balanced', probability=False, random_state=rs)))
    add('KNeighborsClassifier', 'kernel_distance', 4, False, lambda rs, nf, ns, nc: _scaled(KNeighborsClassifier(n_neighbors=max(1, min(int(_default(config, 'knn_neighbors', 5)), ns - 1)))))
    add('GaussianProcessClassifier', 'kernel_distance', 5, False, lambda rs, nf, ns, nc: _scaled(GaussianProcessClassifier(kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.001), random_state=rs)))
    add('DecisionTreeClassifier', 'tree_ensemble', 3, False, lambda rs, nf, ns, nc: DecisionTreeClassifier(max_depth=max_depth, random_state=rs, class_weight='balanced'))
    add('RandomForestClassifier', 'tree_ensemble', 4, True, lambda rs, nf, ns, nc: RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=rs, n_jobs=n_jobs, class_weight='balanced'))
    add('ExtraTreesClassifier', 'tree_ensemble', 4, True, lambda rs, nf, ns, nc: ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=rs, n_jobs=n_jobs, class_weight='balanced'))
    add('GradientBoostingClassifier', 'tree_ensemble', 4, False, lambda rs, nf, ns, nc: GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=rs))
    add('HistGradientBoostingClassifier', 'tree_ensemble', 4, False, lambda rs, nf, ns, nc: HistGradientBoostingClassifier(max_iter=n_estimators, max_leaf_nodes=15, learning_rate=learning_rate, random_state=rs))
    add('AdaBoostClassifier', 'tree_ensemble', 4, False, lambda rs, nf, ns, nc: AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=rs))
    add('BaggingClassifier', 'tree_ensemble', 4, False, lambda rs, nf, ns, nc: _bagging_classifier(rs, n_estimators, n_jobs))
    add('XGBClassifier', 'boosting', 5, True, lambda rs, nf, ns, nc: _xgb_classifier(rs, n_estimators, max_depth, learning_rate, n_jobs, nc), package='xgboost')
    add('LGBMClassifier', 'boosting', 5, True, lambda rs, nf, ns, nc: __import__('lightgbm').LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, num_leaves=min(15, 2 ** max_depth), min_child_samples=5, learning_rate=learning_rate, class_weight='balanced', random_state=rs, n_jobs=n_jobs, verbose=-1), package='lightgbm')
    add('CatBoostClassifier', 'boosting', 5, True, lambda rs, nf, ns, nc: __import__('catboost').CatBoostClassifier(iterations=n_estimators, depth=min(max_depth, 4), learning_rate=learning_rate, loss_function='MultiClass' if nc > 2 else 'Logloss', auto_class_weights='Balanced', random_seed=rs, verbose=False, allow_writing_files=False), package='catboost')
    add('MLPClassifier', 'deep_learning', 6, False, lambda rs, nf, ns, nc: _scaled(MLPClassifier(hidden_layer_sizes=tuple(_default(config, 'mlp_hidden_layer_sizes', [64, 32])), alpha=float(_default(config, 'mlp_alpha', 0.001)), early_stopping=True, validation_fraction=0.2, max_iter=max_iter, random_state=rs)))
    logger.info('Available classification model count: %d', len(specs))
    return (specs, unavailable)
