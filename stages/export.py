from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
CLASSIFICATION = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'pipeline_output' / 'classification'
SUMMARY = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'global_summary'
CONFIRMATION = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / '_no_confirmation'
PAPER = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
FEATURES = ['AD1/AG', 'AD2/AG', 'AD3/AG', 'AD4/AG', 'D Center', 'G Center', 'D/G']
IMAGE_EXTENSIONS = {'.png', '.pdf', '.svg'}

def safe_feature(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', value)

def write_data(stem: Path, data: pd.DataFrame, description: str, sources: list[Path]) -> dict:
    output = stem.with_name(stem.name + '_plot_data.csv')
    data.to_csv(output, index=False, encoding='utf-8-sig')
    return {'figure_stem': str(stem.relative_to(ROOT)).replace('\\', '/'), 'plot_data_csv': str(output.relative_to(ROOT)).replace('\\', '/'), 'description': description, 'source_files': '; '.join((str(path.relative_to(ROOT)).replace('\\', '/') for path in sources)), 'n_rows': len(data), 'n_columns': len(data.columns)}

def reshape_shap(values: pd.DataFrame) -> pd.DataFrame:
    meta = ['target', 'task_id', 'task_name', 'scope', 'grade', 'model', 'feature_set', 'explained_class', 'explained_class_name', 'sample_index_task', 'expected_value']
    parts = []
    for feature in FEATURES:
        part = values[meta].copy()
        part['feature'] = feature
        part['normalized_feature_value'] = values[f'value_{feature}'].to_numpy()
        part['shap_value'] = values[f'shap_{feature}'].to_numpy()
        part['abs_shap_value'] = part['shap_value'].abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)

def main() -> None:
    records = []
    best_path = CLASSIFICATION / 'best_classification_model_summary.csv'
    metrics_path = CLASSIFICATION / 'classification_model_metrics.csv'
    ranking_path = CLASSIFICATION / 'classification_shap_feature_ranking.csv'
    shap_path = CLASSIFICATION / 'classification_shap_values.csv'
    stability_path = CLASSIFICATION / 'classification_shap_stability_across_folds.csv'
    repeat_path = CLASSIFICATION / 'classification_shap_stability_by_repeat.csv'
    best = pd.read_csv(best_path)
    metrics = pd.read_csv(metrics_path)
    ranking = pd.read_csv(ranking_path)
    values = pd.read_csv(shap_path)
    stability = pd.read_csv(stability_path)
    repeats = pd.read_csv(repeat_path)
    overview = best.copy()
    overview['reference_line_BA_080'] = 0.8
    records.append(write_data(CLASSIFICATION / 'figures' / 'best_classification_balanced_accuracy', overview, 'Best-model Task-B balanced accuracy by target, including cross-fold SD and the BA=0.80 reference.', [best_path]))
    model_box = metrics[(metrics['cv_type'] == 'repeated_stratified_kfold') & (metrics['n_success_folds'] > 0)].copy()
    model_box['reference_line_random'] = 0.5
    records.append(write_data(CLASSIFICATION / 'figures' / 'classification_model_balanced_accuracy_boxplot', model_box, 'Per-model mean balanced accuracy used by the classifier comparison boxplot.', [metrics_path]))
    for target in best['target']:
        figure_dir = CLASSIFICATION / 'shap' / target / 'task_B_high_vs_low' / 'main'
        target_rank = ranking[ranking['target'] == target].sort_values('rank').copy()
        target_values = values[values['target'] == target].copy()
        target_long = reshape_shap(target_values)
        target_stability = stability[stability['target'] == target].copy()
        target_repeat = repeats[repeats['target'] == target].copy()
        records.append(write_data(figure_dir / 'classification_shap_bar_plot', target_rank, 'Mean absolute SHAP ranking for the high-performance class.', [ranking_path]))
        records.append(write_data(figure_dir / 'classification_shap_beeswarm_plot', target_long, 'Long-form SHAP values and normalized descriptor values for every explained held-out sample.', [shap_path]))
        stable = target_stability.merge(target_repeat, on=[column for column in ['target', 'task_id', 'feature'] if column in target_repeat.columns], how='left', suffixes=('_summary', '_repeat'))
        records.append(write_data(figure_dir / 'classification_shap_cross_fold_stability', stable, 'Cross-fold and repeat-level SHAP stability statistics used by the stability figure.', [stability_path, repeat_path]))
        present_stems = {path.stem.removeprefix('classification_shap_dependence_') for path in figure_dir.glob('classification_shap_dependence_*.png')}
        for feature in target_rank['feature']:
            if safe_feature(feature) not in present_stems:
                continue
            dependence = target_long[target_long['feature'] == feature].copy()
            dependence['x_normalized_descriptor_value'] = dependence['normalized_feature_value']
            dependence['y_shap_value'] = dependence['shap_value']
            records.append(write_data(figure_dir / f'classification_shap_dependence_{safe_feature(feature)}', dependence, f'Dependence scatter coordinates for normalized {feature}.', [shap_path]))
        displayed = target_values.iloc[0]
        waterfall = pd.DataFrame({'feature': FEATURES, 'normalized_feature_value': [displayed[f'value_{feature}'] for feature in FEATURES], 'shap_contribution': [displayed[f'shap_{feature}'] for feature in FEATURES]})
        waterfall['abs_shap_contribution'] = waterfall['shap_contribution'].abs()
        waterfall['sample_index_task'] = displayed['sample_index_task']
        waterfall['expected_value'] = displayed['expected_value']
        waterfall = waterfall.sort_values('abs_shap_contribution', ascending=False)
        records.append(write_data(figure_dir / 'classification_shap_waterfall_plot', waterfall, 'Normalized feature values, base value and SHAP contributions for the waterfall sample.', [shap_path]))
    summary_best_path = SUMMARY / 'best_models_summary.csv'
    summary_rank_path = SUMMARY / 'shap_feature_ranking.csv'
    summary_stability_path = SUMMARY / 'shap_cross_fold_stability.csv'
    records.append(write_data(SUMMARY / 'figures' / 'best_balanced_accuracy_summary', pd.read_csv(summary_best_path), 'Best-model BA means and SDs in the global summary.', [summary_best_path]))
    records.append(write_data(SUMMARY / 'figures' / 'unified_catboost_shap_importance_heatmap', pd.read_csv(summary_rank_path), 'Long-form target-by-feature mean absolute SHAP values used by the heatmap.', [summary_rank_path]))
    summary_stability = pd.read_csv(summary_stability_path)
    records.append(write_data(SUMMARY / 'figures' / 'ad4_shap_cross_fold_stability', summary_stability[summary_stability['feature'] == 'AD4/AG'], 'AD4/AG SHAP mean, SD and rank data used by the two-panel stability figure.', [summary_stability_path]))
    comparison_path = SUMMARY / 'noVIF_vs_previous_selected_pipeline.csv'
    if comparison_path.exists():
        records.append(write_data(SUMMARY / 'figures' / 'noVIF_vs_previous_selected_pipeline', pd.read_csv(comparison_path), 'Seven-descriptor pruned-99 result versus the earlier five-descriptor reference.', [comparison_path]))
    paper_index = PAPER / 'figure_plot_data_index.csv'
    if paper_index.exists():
        for item in pd.read_csv(paper_index).itertuples(index=False):
            records.append({'figure_stem': item.figure_stem, 'plot_data_csv': item.plot_data_csv, 'description': 'Paper-style target validation figure with adjacent plotting data.', 'source_files': 'paper_results generated from active classification tables', 'n_rows': len(pd.read_csv(ROOT / item.plot_data_csv)), 'n_columns': len(pd.read_csv(ROOT / item.plot_data_csv).columns)})
    index = pd.DataFrame(records).sort_values('figure_stem').reset_index(drop=True)
    index.insert(0, 'figure_id', np.arange(1, len(index) + 1))
    index.to_csv(ROOT / 'figure_plot_data_index.csv', index=False, encoding='utf-8-sig')
    active = [CLASSIFICATION, SUMMARY, CONFIRMATION, PAPER]
    figure_stems = {str(path.with_suffix('').relative_to(ROOT)).replace('\\', '/') for folder in active for path in folder.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    exported = set(index['figure_stem'])
    missing = sorted(figure_stems - exported)
    extra = sorted(exported - figure_stems)
    audit = pd.DataFrame([{'check': 'active_figure_stems', 'count': len(figure_stems), 'details': ''}, {'check': 'plot_data_csv_files', 'count': len(index), 'details': ''}, {'check': 'missing_plot_data', 'count': len(missing), 'details': '; '.join(missing)}, {'check': 'extra_plot_data', 'count': len(extra), 'details': '; '.join(extra)}])
    audit.to_csv(ROOT / 'figure_plot_data_audit.csv', index=False, encoding='utf-8-sig')
    if missing or extra:
        raise RuntimeError(f'Plot-data coverage mismatch. Missing={missing}; extra={extra}')
    print(f'Created {len(index)} plot-data CSV files; all active figures are covered.')
if __name__ == '__main__':
    main()
