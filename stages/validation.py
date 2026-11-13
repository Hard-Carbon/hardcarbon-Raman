from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
ROOT = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
SOURCE = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'pipeline_output' / 'classification'
OUT = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
FORMATS = ('png', 'pdf', 'svg')

def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in FORMATS:
        fig.savefig(stem.with_suffix(f'.{extension}'), dpi=600, bbox_inches='tight')
    plt.close(fig)

def data(stem: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(stem.with_name(stem.name + '_plot_data.csv'), index=False, encoding='utf-8-sig')

def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(['sample_index_task', 'sample_id'], as_index=False).agg(normalized_target_value=('normalized_target_value', 'mean'), true_label=('true_label', 'first'), true_label_name=('true_label_name', 'first'), predicted_label=('predicted_label', lambda values: int(pd.Series(values).mode().iloc[0])), probability_high=('positive_class_score', 'mean'), correct_rate=('correct', 'mean'), n_cv_predictions=('correct', 'size'))

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11, 'svg.fonttype': 'none'})
    best = pd.read_csv(SOURCE / 'best_classification_model_summary.csv')
    metrics = pd.read_csv(SOURCE / 'classification_model_metrics.csv')
    folds = pd.read_csv(SOURCE / 'classification_fold_metrics.csv')
    predictions = pd.read_csv(SOURCE / 'classification_predictions.csv')
    index_rows = []
    for row in best.itertuples(index=False):
        target = row.target
        task = row.task_id
        out = OUT / target
        raw = predictions[(predictions['target'] == target) & (predictions['task_id'] == task) & (predictions['model'] == row.best_model) & (predictions['feature_set'] == row.best_feature_set)].copy()
        agg = aggregate(raw)
        raw.to_csv(out / 'predictions' / 'best_cv_predictions.csv', index=False, encoding='utf-8-sig') if (out / 'predictions').mkdir(parents=True, exist_ok=True) is None else None
        agg.to_csv(out / 'predictions' / 'best_cv_predictions_aggregated.csv', index=False, encoding='utf-8-sig')
        (out / 'best_model').mkdir(parents=True, exist_ok=True)
        (out / 'best_model' / 'best_model.json').write_text(json.dumps({'model': row.best_model, 'feature_set': row.best_feature_set, 'BA_mean': row.BalancedAccuracy_mean, 'BA_SD': row.BalancedAccuracy_std, 'F1_mean': row.F1_mean, 'ROC_AUC_mean': row.ROC_AUC_mean, 'PR_AUC_mean': row.PR_AUC_mean, 'MCC_mean': row.MCC_mean}, indent=2), encoding='utf-8')
        model_data = metrics[(metrics['target'] == target) & (metrics['task_id'] == task) & (metrics['n_success_folds'] > 0)].sort_values('BalancedAccuracy_mean', ascending=False).head(14)
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        x = np.arange(len(model_data))
        ax.bar(x, model_data['BalancedAccuracy_mean'], yerr=model_data['BalancedAccuracy_std'], capsize=2.5, color='#4C78A8')
        ax.axhline(0.5, color='#555555', linestyle='--', linewidth=0.9)
        ax.set_xticks(x, model_data['model'], rotation=65, ha='right')
        ax.set_ylabel('Balanced accuracy (mean +/- SD)')
        ax.set_title(f'{target}: classifier comparison')
        stem = out / 'model_comparison' / 'model_comparison_task_B_high_vs_low'
        save(fig, stem)
        data(stem, model_data)
        index_rows.append(stem)
        labels = [0, 1]
        names = [str(raw.loc[raw['true_label'] == label, 'true_label_name'].iloc[0]) for label in labels]
        matrix = confusion_matrix(raw['true_label'], raw['predicted_label'], labels=labels)
        cm_data = pd.DataFrame(matrix, index=names, columns=names).rename_axis('actual_class').reset_index().melt(id_vars='actual_class', var_name='predicted_class', value_name='count')
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, cbar=False, ax=ax)
        ax.set(xlabel='Predicted class', ylabel='Actual class', title=f'{target}: confusion matrix')
        stem = out / 'model_validation' / 'confusion_matrix_task_B_high_vs_low'
        save(fig, stem)
        data(stem, cm_data)
        index_rows.append(stem)
        fpr, tpr, thresholds = roc_curve(raw['true_label'], raw['positive_class_score'])
        auc = roc_auc_score(raw['true_label'], raw['positive_class_score'])
        roc_data = pd.DataFrame({'false_positive_rate': fpr, 'true_positive_rate': tpr, 'threshold': thresholds, 'ROC_AUC': auc})
        fig, ax = plt.subplots(figsize=(3.9, 3.5))
        ax.plot(fpr, tpr, color='#4C78A8', label=f'ROC-AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='#555555')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{target}: ROC curve')
        ax.legend(frameon=False, loc='lower right')
        stem = out / 'model_validation' / 'roc_curve_task_B_high_vs_low'
        save(fig, stem)
        data(stem, roc_data)
        index_rows.append(stem)
        precision, recall, threshold = precision_recall_curve(raw['true_label'], raw['positive_class_score'])
        ap = average_precision_score(raw['true_label'], raw['positive_class_score'])
        pr_data = pd.DataFrame({'recall': recall, 'precision': precision, 'threshold': np.r_[threshold, np.nan], 'PR_AUC_average_precision': ap})
        fig, ax = plt.subplots(figsize=(3.9, 3.5))
        ax.plot(recall, precision, color='#E19043', label=f'PR-AUC = {ap:.3f}')
        ax.set(xlabel='Recall', ylabel='Precision', title=f'{target}: precision-recall curve')
        ax.legend(frameon=False, loc='lower left')
        stem = out / 'model_validation' / 'pr_curve_task_B_high_vs_low'
        save(fig, stem)
        data(stem, pr_data)
        index_rows.append(stem)
        score_data = agg.copy()
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        sns.scatterplot(data=score_data, x='normalized_target_value', y='probability_high', hue='true_label_name', size='correct_rate', sizes=(32, 88), palette='Set2', ax=ax)
        ax.set(xlabel='Normalized target value', ylabel='Mean high-class probability', title=f'{target}: out-of-fold screening score')
        stem = out / 'predictions' / 'score_vs_normalized_target_task_B_high_vs_low'
        save(fig, stem)
        data(stem, score_data)
        index_rows.append(stem)
        fold = folds[(folds['target'] == target) & (folds['task_id'] == task) & (folds['model'] == row.best_model) & (folds['feature_set'] == row.best_feature_set) & (folds['fit_status'] == 'success')].copy()
        long = fold.melt(id_vars=['repeat', 'fold'], value_vars=['BalancedAccuracy', 'F1', 'MCC'], var_name='metric', value_name='value')
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
        sns.boxplot(data=long, x='metric', y='value', color='#72B7B2', fliersize=1.5, ax=ax)
        ax.axhline(0.5, color='#555555', linestyle='--', linewidth=0.8)
        ax.set(xlabel='', ylabel='Fold metric value', title=f'{target}: CV stability')
        stem = out / 'stability' / 'cv_stability_boxplot_task_B_high_vs_low'
        save(fig, stem)
        data(stem, long)
        index_rows.append(stem)
        fold['split_index'] = np.arange(1, len(fold) + 1)
        fig, ax = plt.subplots(figsize=(5.3, 3.3))
        ax.plot(fold['split_index'], fold['BalancedAccuracy'], color='#4C78A8', linewidth=1.0)
        ax.axhline(fold['BalancedAccuracy'].mean(), color='#111111', linestyle='--')
        ax.set(xlabel='CV split index', ylabel='Balanced accuracy', title=f'{target}: split stability')
        stem = out / 'stability' / 'cv_balanced_accuracy_lineplot_task_B_high_vs_low'
        save(fig, stem)
        data(stem, fold)
        index_rows.append(stem)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.scatter(agg['normalized_target_value'], agg['probability_high'], c=agg['true_label'], cmap='viridis', s=46, edgecolors='white', linewidths=0.4)
        ax.axhline(0.5, color='#555555', linestyle='--')
        ax.set(xlabel='Normalized target value', ylabel='Mean high-class probability', title=f'{target}: performance validation')
        ax.text(0.04, 0.95, f'BA = {row.BalancedAccuracy_mean:.3f} +/- {row.BalancedAccuracy_std:.3f}\nF1 = {row.F1_mean:.3f}\nROC-AUC = {row.ROC_AUC_mean:.3f}\nMCC = {row.MCC_mean:.3f}', transform=ax.transAxes, va='top')
        stem = out / 'model_validation' / 'performance_validation_task_B_high_vs_low'
        save(fig, stem)
        data(stem, agg)
        index_rows.append(stem)
    rows = [{'figure_stem': str(path.relative_to(ROOT)).replace('\\', '/'), 'plot_data_csv': str(path.with_name(path.name + '_plot_data.csv').relative_to(ROOT)).replace('\\', '/')} for path in index_rows]
    pd.DataFrame(rows).to_csv(OUT / 'figure_plot_data_index.csv', index=False, encoding='utf-8-sig')
    (OUT / 'README.md').write_text('# Pruned-99 paper figures\n\nPer-target model comparison, confusion matrix, ROC, PR, score, CV-stability and validation figures. Each figure has an adjacent `*_plot_data.csv`.\n', encoding='utf-8')
    print(f'Created {len(index_rows)} supplementary figure stems.')
if __name__ == '__main__':
    main()
