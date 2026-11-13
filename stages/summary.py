from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
RUN = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'pipeline_output' / 'classification'
OUT = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'global_summary'
FIG = OUT / 'figures'
FEATURES = ['AD1/AG', 'AD2/AG', 'AD3/AG', 'AD4/AG', 'D Center', 'G Center', 'D/G']
INPUT_NORMALIZED = Path(__file__).resolve().parents[1] / 'data' / 'normalized'
PREVIOUS_REFERENCE = Path(__file__).resolve().parents[1] / 'reference' / 'five_feature_summary.csv'

def save_figure(fig, stem):
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(FIG / f'{stem}.{suffix}', dpi=600, bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data_used = ROOT / 'data_used'
    data_used.mkdir(parents=True, exist_ok=True)
    source_dir = INPUT_NORMALIZED
    for target in ('0.2C-R', 'Dp-2', 'Ds-2', 'SL-R'):
        source = pd.read_csv(source_dir / f'{target}.csv')
        source[FEATURES + [target]].to_csv(data_used / f'{target}_seven_features_normalized.csv', index=False, encoding='utf-8-sig')
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'svg.fonttype': 'none'})
    best_path = RUN / 'best_classification_model_summary.csv'
    metrics_path = RUN / 'classification_model_metrics.csv'
    folds_path = RUN / 'classification_fold_metrics.csv'
    ranking_path = RUN / 'classification_shap_feature_ranking.csv'
    stability_path = RUN / 'classification_shap_stability_across_folds.csv'
    reliability_path = RUN / 'classification_reliability_grade.csv'
    unified_path = RUN / 'classification_unified_model_performance.csv'
    best = pd.read_csv(best_path)
    metrics = pd.read_csv(metrics_path)
    folds = pd.read_csv(folds_path)
    ranking = pd.read_csv(ranking_path)
    stability = pd.read_csv(stability_path)
    reliability = pd.read_csv(reliability_path)
    unified = pd.read_csv(unified_path)
    best.to_csv(OUT / 'best_models_summary.csv', index=False, encoding='utf-8-sig')
    metrics.to_csv(OUT / 'all_model_metrics.csv', index=False, encoding='utf-8-sig')
    ranking.to_csv(OUT / 'shap_feature_ranking.csv', index=False, encoding='utf-8-sig')
    stability.to_csv(OUT / 'shap_cross_fold_stability.csv', index=False, encoding='utf-8-sig')
    reliability.to_csv(OUT / 'classification_reliability_summary.csv', index=False, encoding='utf-8-sig')
    unified.to_csv(OUT / 'unified_model_performance.csv', index=False, encoding='utf-8-sig')
    expected = set(FEATURES)
    audit_rows = []
    for target, frame in folds.groupby('target', sort=False):
        exact = frame['selected_features'].fillna('').map(lambda x: set(x.split(';')) == expected)
        audit_rows.append({'target': target, 'vif_enabled': False, 'features': ';'.join(FEATURES), 'n_models': frame['model'].nunique(), 'n_repeat_fold_splits': frame[['repeat', 'fold']].drop_duplicates().shape[0], 'n_fold_model_records': len(frame), 'n_success': int((frame['fit_status'] == 'success').sum()), 'n_failed': int((frame['fit_status'] != 'success').sum()), 'min_n_features': frame['n_features'].min(), 'max_n_features': frame['n_features'].max(), 'all_records_use_exact_seven_features': bool(exact.all())})
    pd.DataFrame(audit_rows).to_csv(OUT / 'feature_usage_audit.csv', index=False, encoding='utf-8-sig')
    ordered = best.sort_values('BalancedAccuracy_mean').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    bars = ax.barh(ordered['target'], ordered['BalancedAccuracy_mean'], xerr=ordered['BalancedAccuracy_std'], color='#2878B5', capsize=3)
    ax.axvline(0.8, color='#C82423', linestyle='--', linewidth=1, label='BA = 0.80')
    ax.set(xlim=(0.45, 1.04), xlabel='Balanced accuracy (mean +/- SD)', ylabel='Target', title='Task B best classifiers: 99 samples, seven raw descriptors, VIF disabled')
    for bar, model in zip(bars, ordered['best_model']):
        ax.text(0.47, bar.get_y() + bar.get_height() / 2, model, va='center', fontsize=7.5)
    ax.legend(frameon=False, loc='lower right')
    save_figure(fig, 'best_balanced_accuracy_summary')
    model_name = ranking['model'].mode().iloc[0]
    heat = ranking.pivot(index='target', columns='feature', values='mean_abs_shap')
    heat = heat.reindex(index=best['target'], columns=FEATURES)
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    sns.heatmap(heat, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Mean |SHAP|'}, ax=ax)
    ax.set(xlabel='Normalized Raman descriptor', ylabel='Target', title=f'Unified {model_name} SHAP importance (VIF disabled)')
    ax.tick_params(axis='x', rotation=30)
    save_figure(fig, 'unified_catboost_shap_importance_heatmap')
    ad4 = stability[stability['feature'] == 'AD4/AG'].copy()
    ad4 = ad4.set_index('target').reindex(best['target']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    axes[0].bar(ad4['target'], ad4['mean_abs_shap_mean'], yerr=ad4['mean_abs_shap_std'], color='#F28E2B', capsize=3)
    axes[0].set(ylabel='Cross-fold mean |SHAP| +/- SD', title='AD4/AG importance')
    axes[1].bar(ad4['target'], ad4['rank_mean'], color='#59A14F')
    axes[1].invert_yaxis()
    axes[1].set(ylim=(7.5, 0.5), ylabel='Mean SHAP rank (1 = highest)', title='AD4/AG rank')
    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save_figure(fig, 'ad4_shap_cross_fold_stability')
    previous_path = PREVIOUS_REFERENCE
    if previous_path.exists():
        previous = pd.read_csv(previous_path)[['target', 'best_model', 'BalancedAccuracy_mean', 'BalancedAccuracy_std']]
        previous = previous.rename(columns={'best_model': 'five_feature_model', 'BalancedAccuracy_mean': 'five_feature_BA_mean', 'BalancedAccuracy_std': 'five_feature_BA_SD'})
        current = best[['target', 'best_model', 'BalancedAccuracy_mean', 'BalancedAccuracy_std']].rename(columns={'best_model': 'seven_feature_model', 'BalancedAccuracy_mean': 'seven_feature_BA_mean', 'BalancedAccuracy_std': 'seven_feature_BA_SD'})
        comparison = current.merge(previous, on='target')
        comparison['seven_minus_five_BA'] = comparison['seven_feature_BA_mean'] - comparison['five_feature_BA_mean']
        comparison.to_csv(OUT / 'noVIF_vs_previous_selected_pipeline.csv', index=False, encoding='utf-8-sig')
        x = np.arange(len(comparison))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.bar(x - width / 2, comparison['five_feature_BA_mean'], width, yerr=comparison['five_feature_BA_SD'], color='#A0A0A0', capsize=2, label='Five features')
        ax.bar(x + width / 2, comparison['seven_feature_BA_mean'], width, yerr=comparison['seven_feature_BA_SD'], color='#2878B5', capsize=2, label='Seven features')
        ax.axhline(0.8, color='#C82423', linestyle='--', linewidth=1)
        ax.set(xticks=x, xticklabels=comparison['target'], ylim=(0.45, 1.04), xlabel='Target', ylabel='Balanced accuracy (mean +/- SD)', title='No-VIF selected-pipeline comparison')
        ax.legend(frameon=False, loc='lower right')
        save_figure(fig, 'noVIF_vs_previous_selected_pipeline')
    (OUT / 'README.md').write_text(f'Task B classification using 99 internally screened samples and seven normalized Raman descriptors. VIF is disabled. Validation is repeated stratified 5-fold CV with 20 repeats. Error bars are fold-level SD. The unified SHAP model selected by the pipeline is {model_name}.\n', encoding='utf-8')
    print(f'Summary written to {OUT}')
if __name__ == '__main__':
    main()
