from pathlib import Path
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
PAPER = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
PIPE = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'pipeline_output' / 'classification'
FEATURES = ['AD1/AG', 'AD2/AG', 'AD3/AG', 'AD4/AG', 'D Center', 'G Center', 'D/G']
TARGETS = ['0.2C-R', 'Dp-2', 'Ds-2', 'SL-R']
FORMATS = ['png', 'pdf', 'svg']
SEED = 2026
REPEATS = 20
SPLITS = 5
RECORDS = []

def safe(name):
    return re.sub('[^A-Za-z0-9_.-]+', '_', name)

def model(seed):
    return RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=seed, n_jobs=-1)

def explain(estimator, x):
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        result = np.asarray(values[1])
    else:
        result = np.asarray(values)
        if result.ndim == 3:
            result = result[:, :, 1]
    base = float(np.asarray(explainer.expected_value).ravel()[1])
    probability = estimator.predict_proba(x)[:, 1]
    np.testing.assert_allclose(base + result.sum(axis=1), probability, atol=1e-07)
    return (result, base, probability)

def long_table(frame, indices, target, values, base, probability):
    parts = []
    for i, feature in enumerate(FEATURES):
        parts.append(pd.DataFrame({'target': target, 'data_row_1based': np.asarray(indices) + 1, 'feature': feature, 'normalized_feature_value': frame.iloc[indices][feature].to_numpy(), 'normalized_target_value': frame.iloc[indices][target].to_numpy(), 'shap_value': values[:, i], 'abs_shap_value': np.abs(values[:, i]), 'expected_value': base, 'probability_high': probability}))
    return pd.concat(parts, ignore_index=True)

def save(fig, stem, data, kind, target, scope):
    stem.parent.mkdir(parents=True, exist_ok=True)
    data_path = stem.with_name(stem.name + '_plot_data.csv')
    data.to_csv(data_path, index=False, encoding='utf-8-sig')
    fig.tight_layout()
    for extension in FORMATS:
        fig.savefig(stem.parent / (stem.name + '.' + extension), dpi=600, bbox_inches='tight')
    plt.close(fig)
    RECORDS.append({'target': target, 'scope': scope, 'plot_type': kind, 'figure_stem': str(stem.relative_to(PAPER)).replace('\\', '/'), 'plot_data_csv': str(data_path.relative_to(PAPER)).replace('\\', '/'), 'n_rows': len(data)})

def axis_plot(ax, part, feature, kind):
    y = 'shap_value' if kind == 'shap_dependence' else 'normalized_target_value'
    if kind == 'shap_dependence':
        dots = ax.scatter(part.normalized_feature_value, part[y], c=part.normalized_target_value, cmap='viridis', s=27, alpha=0.78, edgecolors='white', linewidths=0.25)
        ax.axhline(0, color='#888888', linestyle='--', linewidth=0.75)
    else:
        dots = ax.scatter(part.normalized_feature_value, part[y], s=27, color='#2878B5', alpha=0.75, edgecolors='white', linewidths=0.25)
    ax.set_xlabel(f'{feature}\n(normalized descriptor value)')
    ax.set_ylabel('SHAP value for high class' if kind == 'shap_dependence' else 'Normalized target value')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.16)
    return dots

def plots(table, target, scope):
    out = PAPER / target / 'dependence' / scope
    for kind in ['feature_target', 'shap_dependence']:
        fig_panel, axes = plt.subplots(2, 4, figsize=(14, 7))
        for i, feature in enumerate(FEATURES):
            part = table[table.feature.eq(feature)].copy()
            fig, ax = plt.subplots(figsize=(4.9, 4.2))
            dots = axis_plot(ax, part, feature, kind)
            ax.set_title(f"{target} | {scope.replace('_', ' ')}\nn = {len(part)}")
            if kind == 'shap_dependence':
                fig.colorbar(dots, ax=ax, label='Normalized target value')
            save(fig, out / f'{kind}_{safe(feature)}', part, kind, target, scope)
            axis_plot(axes.ravel()[i], part, feature, kind)
        axes.ravel()[-1].axis('off')
        fig_panel.suptitle(f"{target} | {scope.replace('_', ' ')} | {kind.replace('_', ' ')}", fontsize=13)
        save(fig_panel, out / f'{kind}_all_features', table, kind + '_panel', target, scope)

def main():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'svg.fonttype': 'none', 'axes.linewidth': 1})
    config = yaml.safe_load((PIPE / 'config_used.yaml').read_text(encoding='utf-8'))
    cv = config['classification_settings']['cv_settings']
    assert (cv['n_splits'], cv['n_repeats'], cv['random_seed']) == (SPLITS, REPEATS, SEED)
    unified = pd.read_csv(PIPE / 'classification_unified_model_performance.csv')
    assert unified.selected_unified_model.eq('RandomForestClassifier').all()
    saved_predictions = pd.read_csv(PIPE / 'classification_predictions.csv')
    label_table = pd.read_csv(PIPE / 'classification_labels.csv')
    summary = []
    for target in TARGETS:
        frame = pd.read_csv(PAPER / 'data_used' / f'{target}_seven_features_normalized.csv')
        assert len(frame) == 99
        assert np.isfinite(frame[FEATURES + [target]].to_numpy()).all()
        low, high = frame[target].quantile([0.25, 0.75])
        chosen = np.flatnonzero(((frame[target] <= low) | (frame[target] >= high)).to_numpy())
        x = frame.iloc[chosen][FEATURES].reset_index(drop=True)
        labels = (frame.iloc[chosen][target].to_numpy() >= high).astype(int)
        saved_labels = label_table[(label_table.target == target) & label_table.included_in_task].sort_values('sample_index')
        np.testing.assert_array_equal(chosen, saved_labels.sample_index.to_numpy())
        np.testing.assert_array_equal(labels, saved_labels.label.to_numpy())
        estimator = model(SEED).fit(x, labels)
        full_values, full_base, full_probability = explain(estimator, frame[FEATURES])
        full = long_table(frame, np.arange(99), target, full_values, full_base, full_probability)
        full['scope'] = 'full99_descriptive'
        rows = []
        for split_no, (train, test) in enumerate(RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=SEED).split(x, labels)):
            repeat, fold = divmod(split_no, SPLITS)
            fold_model = model(SEED + 1000 * repeat + fold).fit(x.iloc[train], labels[train])
            values, base, probability = explain(fold_model, x.iloc[test])
            pred = saved_predictions[(saved_predictions.target == target) & (saved_predictions.model == 'RandomForestClassifier') & (saved_predictions.repeat == repeat) & (saved_predictions.fold == fold)].sort_values('sample_index_task')
            np.testing.assert_array_equal(pred.sample_index_task.to_numpy(), np.sort(test))
            np.testing.assert_allclose(pred.positive_class_score.to_numpy(), probability[np.argsort(test)], atol=1e-10)
            part = long_table(frame, chosen[test], target, values, base, probability)
            part['repeat'] = repeat
            part['fold'] = fold
            part['scope'] = 'heldout_fold'
            rows.append(part)
        heldout = pd.concat(rows, ignore_index=True)
        oof = heldout.groupby(['target', 'data_row_1based', 'feature'], as_index=False).agg(normalized_feature_value=('normalized_feature_value', 'first'), normalized_target_value=('normalized_target_value', 'first'), shap_value=('shap_value', 'mean'), shap_sd=('shap_value', 'std'), mean_abs_shap=('abs_shap_value', 'mean'), expected_value=('expected_value', 'mean'), probability_high=('probability_high', 'mean'), n_heldout_predictions=('repeat', 'size'))
        assert oof.n_heldout_predictions.eq(REPEATS).all()
        assert len(oof) == 50 * len(FEATURES)
        for scope, table in [('full99', full), ('test_oof50', oof)]:
            out = PAPER / target / 'dependence' / scope
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / 'all_features_long.csv', index=False, encoding='utf-8-sig')
            plots(table, target, scope)
        heldout.to_csv(PAPER / target / 'dependence' / 'test_oof50' / 'shap_heldout_all_folds_long.csv', index=False, encoding='utf-8-sig')
        summary.append({'target': target, 'full_samples': 99, 'test_unique_samples': len(chosen), 'heldout_predictions': len(heldout) // len(FEATURES), 'features': 7, 'prediction_match_verified': True, 'shap_additivity_verified': True})
        print(f'Completed {target}: full99 and heldout50, prediction and SHAP checks passed.', flush=True)
    pd.DataFrame(RECORDS).to_csv(PAPER / 'dependence_plot_index.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(PAPER / 'dependence_verification.csv', index=False, encoding='utf-8-sig')
    index = []
    for stem in sorted({p.with_suffix('') for p in PAPER.rglob('*') if p.suffix.lower() in {'.png', '.pdf', '.svg'}}):
        csv = stem.with_name(stem.name + '_plot_data.csv')
        index.append({'figure_stem': str(stem.relative_to(PAPER)).replace('\\', '/'), 'plot_data_csv': str(csv.relative_to(PAPER)).replace('\\', '/'), 'plot_data_exists': csv.exists()})
    index = pd.DataFrame(index)
    index.to_csv(PAPER / 'all_figure_plot_data_index.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'check': 'unique_figure_stems', 'count': len(index)}, {'check': 'missing_adjacent_plot_data', 'count': int((~index.plot_data_exists).sum())}]).to_csv(PAPER / 'all_figure_plot_data_audit.csv', index=False, encoding='utf-8-sig')
    (PAPER / 'DEPENDENCE_README.md').write_text('# Descriptor dependence plots\n\nEach target/dependence folder contains full99 and test_oof50. Both contain all seven feature-target scatter plots and all seven SHAP dependence plots, plus seven-feature panels. PNG/PDF/SVG are exported at 600 dpi and every figure has an adjacent plotting CSV.\n\nfull99: descriptor versus measured target uses all 99 observations. For SHAP, the existing Task-B RandomForestClassifier is refitted on the 50 high/low observations and explains all 99 observations. The 49 middle observations are descriptive model extrapolation within the observed descriptor domain, not classification validation.\n\ntest_oof50: only high/low Task-B observations have held-out predictions. Each sample is evaluated once per repeat, using exactly the original 5-fold x20 splits and RF parameters/seeds. The displayed 50-point SHAP scatter averages signed held-out contributions per sample; the CSV also reports SD and mean absolute SHAP separately. Global importance must use mean_abs_shap, not abs(mean signed SHAP).\n\nshap_heldout_all_folds_long.csv records all 1000 held-out predictions per target and feature, with repeat/fold IDs. Feature-target scatter plots use unique observations, avoiding repeated copies of identical measured points.\n\nThe existing model selection and reported metrics are unchanged. No independent held-out test set was available; test here means cross-validation held-out folds.\n', encoding='utf-8')
    assert index.plot_data_exists.all(), 'A figure is missing plot data.'
    print(json.dumps({'new_figure_stems': len(RECORDS), 'total_figure_stems': len(index), 'missing_plot_data': 0}), flush=True)
if __name__ == '__main__':
    main()
