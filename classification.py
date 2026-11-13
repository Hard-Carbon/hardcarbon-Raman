from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.classification_mode import run_classification_workflow
from src.classification_model_zoo import build_classification_model_zoo
from src.data_loader import collect_excluded_target_rows, load_normalized_datasets
from src.utils import copy_config, deep_update, ensure_project_dirs, load_config, sanitize_name, save_table, set_global_seed, setup_logger
from src.visualization import save_figure, set_plot_style

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AI4S normalized-data classification screening workflow.')
    parser.add_argument('--config', default='config_run_classification_nor.yaml', help='Path to classification config YAML.')
    parser.add_argument('--smoke-test', action='store_true', help='Run a short syntax/runtime validation pass.')
    return parser.parse_args()

def apply_smoke_overrides(config: dict) -> dict:
    overrides = {'output_dir': 'results_classification_smoke', 'log_dir': 'logs_classification_smoke', 'classification_settings': {'cv_settings': {'n_splits': 3, 'n_repeats': 1, 'random_seed': 42}, 'enabled_models': ['LogisticRegression', 'ExtraTreesClassifier', 'SVC_rbf'], 'feature_sets': ['raw_feature_set', 'vif_filtered_feature_set'], 'shap_enabled': False, 'defaults': {'n_estimators': 30, 'max_iter': 300}}}
    return deep_update(config, overrides)

def build_target_normalization_check(datasets: dict) -> pd.DataFrame:
    rows = []
    for target, dataset in datasets.items():
        feature_min = dataset.X.min(numeric_only=True).min()
        feature_max = dataset.X.max(numeric_only=True).max()
        y = dataset.y.astype(float)
        rows.append({'target': target, 'source_file': str(dataset.source_path), 'n_samples': len(dataset.y), 'n_features': len(dataset.feature_columns), 'target_column': dataset.target_column, 'target_min': float(y.min()), 'target_max': float(y.max()), 'target_mean': float(y.mean()), 'target_std': float(y.std(ddof=1)), 'target_within_0_1': bool(y.min() >= -1e-09 and y.max() <= 1 + 1e-09), 'all_feature_min': float(feature_min), 'all_feature_max': float(feature_max), 'all_features_within_0_1': bool(feature_min >= -1e-09 and feature_max <= 1 + 1e-09)})
    return pd.DataFrame(rows)

def save_classification_results(results, classification_dir: Path) -> None:
    save_table(results.labels, classification_dir / 'classification_labels.csv')
    save_table(results.metrics, classification_dir / 'classification_model_metrics.csv')
    save_table(results.fold_metrics, classification_dir / 'classification_fold_metrics.csv')
    save_table(results.best_summary, classification_dir / 'best_classification_model_summary.csv')
    save_table(results.model_selection_scores, classification_dir / 'classification_model_selection_scores.csv')
    save_table(results.predictions, classification_dir / 'classification_predictions.csv')
    save_table(results.confusion_matrices, classification_dir / 'classification_confusion_matrices.csv')
    save_table(results.roc_curves, classification_dir / 'classification_roc_curves.csv')
    save_table(results.pr_curves, classification_dir / 'classification_pr_curves.csv')
    save_table(results.unified_selection, classification_dir / 'classification_unified_model_selection.csv')
    save_table(results.unified_performance, classification_dir / 'classification_unified_model_performance.csv')
    save_table(results.shap_values, classification_dir / 'classification_shap_values.csv')
    save_table(results.shap_feature_ranking, classification_dir / 'classification_shap_feature_ranking.csv')
    save_table(results.shap_stability, classification_dir / 'classification_shap_stability_across_folds.csv')
    save_table(results.shap_stability_by_repeat, classification_dir / 'classification_shap_stability_by_repeat.csv')
    save_table(results.reliability, classification_dir / 'classification_reliability_grade.csv')

def make_regression_classification_context(root: Path, results, classification_dir: Path) -> None:
    regression_path = root / 'results_core_run' / 'metrics' / 'best_model_summary.csv'
    if not regression_path.exists() or results.best_summary.empty:
        return
    regression = pd.read_csv(regression_path)
    keep_cols = [col for col in ['target', 'best_model', 'R2_mean', 'MAE_mean', 'RMSE_mean'] if col in regression.columns]
    if 'target' not in keep_cols:
        return
    merged = results.best_summary.merge(regression[keep_cols], on='target', how='left', suffixes=('_classification', '_regression'))
    merged = merged.rename(columns={'best_model_classification': 'best_classification_model', 'best_model_regression': 'best_regression_model', 'R2_mean': 'regression_R2_mean', 'MAE_mean': 'regression_MAE_mean', 'RMSE_mean': 'regression_RMSE_mean'})
    merged['interpretation_hint'] = merged.apply(_classification_interpretation_hint, axis=1)
    save_table(merged, classification_dir / 'classification_vs_regression_context.csv')

def _classification_interpretation_hint(row: pd.Series) -> str:
    r2 = row.get('regression_R2_mean')
    ba = row.get('BalancedAccuracy_mean')
    grade = row.get('grade')
    if pd.notna(r2) and pd.notna(ba) and (r2 < 0.2) and (ba >= 0.65):
        return 'Low regression R2 but useful classification signal for high-performance screening.'
    if grade in {'A', 'B'}:
        return 'Classification result is suitable for main or supporting discussion.'
    if grade == 'D':
        return 'Classification is near chance; avoid strong mechanistic claims.'
    return 'Exploratory classification result.'

def organize_by_target_task(results, classification_dir: Path) -> None:
    by_dir = classification_dir / 'by_target_task'
    table_specs = [('classification_labels.csv', results.labels), ('classification_model_metrics.csv', results.metrics), ('classification_fold_metrics.csv', results.fold_metrics), ('best_classification_model_summary.csv', results.best_summary), ('classification_model_selection_scores.csv', results.model_selection_scores), ('classification_predictions.csv', results.predictions), ('classification_confusion_matrices.csv', results.confusion_matrices), ('classification_roc_curves.csv', results.roc_curves), ('classification_pr_curves.csv', results.pr_curves), ('classification_unified_model_performance.csv', results.unified_performance), ('classification_shap_values.csv', results.shap_values), ('classification_shap_feature_ranking.csv', results.shap_feature_ranking), ('classification_shap_stability_across_folds.csv', results.shap_stability), ('classification_shap_stability_by_repeat.csv', results.shap_stability_by_repeat), ('classification_reliability_grade.csv', results.reliability)]
    for filename, table in table_specs:
        if table is None or table.empty or 'target' not in table.columns:
            continue
        group_cols = ['target', 'task_id'] if 'task_id' in table.columns else ['target']
        for keys, group in table.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys, '')
            target = sanitize_name(keys[0])
            task = sanitize_name(keys[1]) if len(keys) > 1 and keys[1] else '_target_summary'
            save_table(group, by_dir / target / task / filename)

def make_classification_overview_figures(results, classification_dir: Path, config: dict) -> None:
    if results.best_summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return
    set_plot_style(config)
    figures_dir = classification_dir / 'figures'
    plot_cfg = config.get('plotting_settings', {})
    data = results.best_summary.copy()
    data['target_task'] = data['target'].astype(str) + '\n' + data['task_name'].astype(str)
    fig, ax = plt.subplots(figsize=(plot_cfg.get('wide_figure_width', 7.0), plot_cfg.get('wide_figure_height', 4.2)))
    sns.barplot(data=data, x='target_task', y='BalancedAccuracy_mean', hue='grade' if 'grade' in data.columns else None, ax=ax)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.9)
    ax.axhline(0.7, color='#555555', linestyle=':', linewidth=0.9)
    ax.set_xlabel('classification task')
    ax.set_ylabel('balanced accuracy')
    ax.tick_params(axis='x', rotation=65)
    if ax.get_legend() is not None:
        ax.legend(title='grade', frameon=False)
    save_figure(fig, figures_dir / 'best_classification_balanced_accuracy', config)
    plt.close(fig)
    if not results.metrics.empty:
        primary_cv = config.get('classification_settings', {}).get('primary_cv_type', 'repeated_stratified_kfold')
        metric_data = results.metrics[(results.metrics['cv_type'] == primary_cv) & (results.metrics['n_success_folds'] > 0)].copy()
        if not metric_data.empty:
            fig, ax = plt.subplots(figsize=(plot_cfg.get('wide_figure_width', 7.0), plot_cfg.get('wide_figure_height', 4.2)))
            sns.boxplot(data=metric_data, x='model', y='BalancedAccuracy_mean', hue='feature_set', ax=ax, fliersize=1.5)
            ax.axhline(0.5, color='black', linestyle='--', linewidth=0.9)
            ax.set_xlabel('classifier')
            ax.set_ylabel('mean balanced accuracy')
            ax.tick_params(axis='x', rotation=65)
            ax.legend(title='feature set', frameon=False)
            save_figure(fig, figures_dir / 'classification_model_balanced_accuracy_boxplot', config)
            plt.close(fig)

def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    if args.smoke_test:
        config = apply_smoke_overrides(config)
    output_dir = ensure_project_dirs(root, config)
    classification_dir = output_dir / 'classification'
    classification_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(root, config)
    seed = int(config.get('classification_settings', {}).get('cv_settings', config.get('cv_settings', {})).get('random_seed', 42))
    set_global_seed(seed)
    copy_config(config, output_dir / 'config_used.yaml')
    copy_config(config, classification_dir / 'config_used.yaml')
    logger.info('Starting AI4S classification screening workflow.')
    logger.info('Classification main data source: %s', config.get('normalized_data_path'))
    datasets = load_normalized_datasets(root, config, logger)
    excluded_target_rows = collect_excluded_target_rows(datasets)
    save_table(excluded_target_rows, classification_dir / 'excluded_rows_non_numeric_target.csv')
    target_check = build_target_normalization_check(datasets)
    save_table(target_check, classification_dir / 'classification_target_normalization_check.csv')
    model_specs, unavailable = build_classification_model_zoo(config, logger)
    save_table(pd.DataFrame(unavailable), classification_dir / 'skipped_models.csv')
    results = run_classification_workflow(datasets, model_specs, root, config, logger)
    if not results.reliability.empty and (not results.best_summary.empty):
        results.best_summary = results.best_summary.merge(results.reliability[['target', 'task_id', 'grade', 'recommended_usage']], on=['target', 'task_id'], how='left')
    save_classification_results(results, classification_dir)
    make_regression_classification_context(root, results, classification_dir)
    organize_by_target_task(results, classification_dir)
    make_classification_overview_figures(results, classification_dir, config)
    logger.info('Classification workflow finished. Results saved to %s', classification_dir)
    print(f'Done. Classification results saved to: {classification_dir}')
if __name__ == '__main__':
    main()
