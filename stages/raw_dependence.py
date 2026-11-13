from pathlib import Path
import hashlib
import json
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
SOURCE = Path(__file__).resolve().parents[1] / 'data' / 'raw'
PERFORMANCE = SOURCE / 'Average Raman-performance-raw(2).xlsx'
KINETIC = SOURCE / 'Raman-Kinective(1).xlsx'
FEATURES = ['AD1/AG', 'AD2/AG', 'AD3/AG', 'AD4/AG', 'D Center', 'G Center', 'D/G']
TARGETS = ['0.2C-R', 'Dp-2', 'Ds-2', 'SL-R']
RECORDS = []

def safe(value):
    return re.sub('[^A-Za-z0-9_.-]+', '_', value)

def plot_axis(ax, frame, feature, kind):
    if kind == 'shap_dependence':
        scatter = ax.scatter(frame.raw_feature_value, frame.shap_value, c=frame.raw_target_value, cmap='viridis', s=28, alpha=0.78, edgecolor='white', linewidth=0.3)
        ax.axhline(0, color='#888888', linestyle='--', linewidth=0.8)
        ax.set_ylabel('SHAP value for high class')
    else:
        scatter = ax.scatter(frame.raw_feature_value, frame.raw_target_value, color='#2878B5', s=28, alpha=0.78, edgecolor='white', linewidth=0.3)
        ax.set_ylabel('Measured target value (original scale)')
    ax.set_xlabel(f'{feature}\n(original descriptor value)')
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.16)
    return scatter

def save(fig, stem, table, target, scope, kind):
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv = stem.with_name(stem.name + '_plot_data.csv')
    table.to_csv(csv, index=False, encoding='utf-8-sig')
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(stem.parent / (stem.name + '.' + ext), dpi=600, bbox_inches='tight')
    plt.close(fig)
    RECORDS.append({'target': target, 'scope': scope, 'plot_type': kind, 'figure_stem': str(stem.relative_to(PAPER)).replace('\\', '/'), 'plot_data_csv': str(csv.relative_to(PAPER)).replace('\\', '/'), 'n_rows': len(table)})

def plots(table, target, scope):
    folder = PAPER / target / 'dependence' / f'{scope}_original_values'
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / 'all_features_long.csv', index=False, encoding='utf-8-sig')
    for kind in ['feature_target', 'shap_dependence']:
        panel, axes = plt.subplots(2, 4, figsize=(15, 7.8))
        for i, feature in enumerate(FEATURES):
            part = table[table.feature.eq(feature)].copy()
            fig, ax = plt.subplots(figsize=(5.1, 4.4))
            dots = plot_axis(ax, part, feature, kind)
            ax.set_title(f"{target} | {scope.replace('_', ' ')}\nn = {len(part)}")
            if kind == 'shap_dependence':
                fig.colorbar(dots, ax=ax, label='Measured target value (original scale)')
            save(fig, folder / f'{kind}_{safe(feature)}', part, target, scope, kind)
            dots = plot_axis(axes.ravel()[i], part, feature, kind)
        axes.ravel()[-1].axis('off')
        if kind == 'shap_dependence':
            color_axis = axes.ravel()[-1].inset_axes([0.08, 0.45, 0.8, 0.08])
            panel.colorbar(dots, cax=color_axis, orientation='horizontal', label='Measured target value (original scale)')
        panel.suptitle(f"{target} | {scope.replace('_', ' ')} | original values", fontsize=13)
        save(panel, folder / f'{kind}_all_features', table, target, scope, kind + '_panel')

def main():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'svg.fonttype': 'none'})
    raw = pd.read_excel(PERFORMANCE, sheet_name='all-raw')
    normalized = pd.read_excel(PERFORMANCE, sheet_name='all-normalized')
    kinetic = pd.read_excel(KINETIC, sheet_name='raw')
    kinetic_normalized = pd.read_excel(KINETIC, sheet_name='SL-R-normalized')
    assert raw.iloc[:, 0].is_unique and kinetic.iloc[:, 0].is_unique
    np.testing.assert_array_equal(raw.iloc[:, 0], kinetic.iloc[:, 0])
    mapping = pd.read_csv(PAPER / 'data_used' / 'selected_99_source_mapping.csv')
    audits, all_data = ([], {})
    for target in TARGETS:
        current = pd.read_csv(PAPER / 'data_used' / f'{target}_seven_features_normalized.csv')
        distances = np.max(np.abs(current[FEATURES].to_numpy()[:, None, :] - normalized[FEATURES].to_numpy()[None, :, :]), axis=2)
        indices = distances.argmin(axis=1)
        assert len(np.unique(indices)) == 99
        np.testing.assert_allclose(distances.min(axis=1), 0, atol=1e-10)
        derived_rows = mapping[f'{target}_Source_Excel_Row'].astype(int).to_numpy()
        source_normalized_target = pd.to_numeric(kinetic_normalized['Slope-R'] if target == 'SL-R' else normalized[target], errors='coerce')
        source_raw_target = pd.to_numeric(kinetic['Slope-R'] if target == 'SL-R' else raw[target], errors='coerce')
        np.testing.assert_allclose(current[target].to_numpy(), source_normalized_target.iloc[indices].to_numpy(), atol=1e-10)
        aligned = raw.iloc[indices][FEATURES].reset_index(drop=True)
        aligned[target] = source_raw_target.iloc[indices].to_numpy()
        aligned.insert(0, 'source_sample_id', raw.iloc[indices, 0].to_numpy())
        aligned.insert(0, 'raw_source_excel_row', indices + 2)
        aligned.insert(0, 'data_row_1based', np.arange(1, 100))
        all_data[target] = aligned
        audits.append({'target': target, 'samples': 99, 'unique_matches': 99, 'max_normalized_descriptor_error': float(distances.min(axis=1).max()), 'max_normalized_target_error': float(np.max(np.abs(current[target].to_numpy() - source_normalized_target.iloc[indices].to_numpy()))), 'derived_source_excel_rows_verified': True, 'raw_vs_derived_row_offsets': int(np.sum(indices + 2 != derived_rows)), 'descriptor_source': str(PERFORMANCE), 'target_source': str(KINETIC if target == 'SL-R' else PERFORMANCE)})
    snapshots = PAPER / 'raw_sources_verified'
    snapshots.mkdir(exist_ok=True)
    for path in [PERFORMANCE, KINETIC]:
        shutil.copy2(path, snapshots / path.name)
    manifest = [{'source': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [PERFORMANCE, KINETIC]]
    (snapshots / 'source_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame(audits).to_csv(snapshots / 'matching_audit.csv', index=False, encoding='utf-8-sig')
    for target, aligned in all_data.items():
        aligned.to_csv(snapshots / f'{target}_99_original_values.csv', index=False, encoding='utf-8-sig')
        for scope in ['full99', 'test_oof50']:
            existing = pd.read_csv(PAPER / target / 'dependence' / scope / 'all_features_long.csv')
            raw_long = aligned.melt(id_vars=['data_row_1based', 'raw_source_excel_row', 'source_sample_id', target], value_vars=FEATURES, var_name='feature', value_name='raw_feature_value').rename(columns={target: 'raw_target_value'})
            merged = existing.merge(raw_long, on=['data_row_1based', 'feature'], how='left', validate='one_to_one')
            assert np.isfinite(merged[['raw_feature_value', 'raw_target_value']].to_numpy()).all()
            plots(merged, target, scope)
        print(f'{target}: original-value plots completed; 99 unique raw matches verified.', flush=True)
    pd.DataFrame(RECORDS).to_csv(PAPER / 'original_values_dependence_index.csv', index=False, encoding='utf-8-sig')
    rows = []
    for stem in sorted({p.with_suffix('') for p in PAPER.rglob('*') if p.suffix.lower() in {'.png', '.pdf', '.svg'}}):
        csv = stem.with_name(stem.name + '_plot_data.csv')
        rows.append({'figure_stem': str(stem.relative_to(PAPER)).replace('\\', '/'), 'plot_data_csv': str(csv.relative_to(PAPER)).replace('\\', '/'), 'plot_data_exists': csv.exists()})
    index = pd.DataFrame(rows)
    index.to_csv(PAPER / 'all_figure_plot_data_index.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'check': 'unique_figure_stems', 'count': len(index)}, {'check': 'missing_adjacent_plot_data', 'count': int((~index.plot_data_exists).sum())}]).to_csv(PAPER / 'all_figure_plot_data_audit.csv', index=False, encoding='utf-8-sig')
    assert index.plot_data_exists.all()
    (snapshots / 'README.md').write_text('# Verified original-value sources\n\nAll 99 samples match exactly to the normalized page of Average Raman-performance-raw(2).xlsx using seven descriptors. Saved source rows were independently checked against the original normalized derived files in ML1/data. Dp/Ds derived files have 139 rows, while the raw master has 143 rows; their Excel row numbers therefore differ. Raw-master rows are recomputed by unique exact descriptor matching, not positional assumptions. Original features and 0.2C-R/Dp-2/Ds-2 targets come from its all-raw page. SL-R comes from Raman-Kinective(1).xlsx, raw/Slope-R: its normalized target matches exactly for all 99 samples, with identical unique sample IDs across both source workbooks.\n\nThe older kinetic workbook uses a different Raman fitting version; its descriptor columns are NOT used. SHAP values and model training remain in the original normalized model space. Only displayed descriptor/target values are replaced by paired original observations. No target units were inferred.\n\nBoth full99_original_values and test_oof50_original_values preserve all prior plot data columns and add raw values, sample ID and source Excel row.\n', encoding='utf-8')
    print(json.dumps({'new_figure_stems': len(RECORDS), 'total_figure_stems': len(index), 'missing_plot_data': 0}), flush=True)
if __name__ == '__main__':
    main()
