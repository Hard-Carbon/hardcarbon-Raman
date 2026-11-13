import csv
import math
from pathlib import Path
PAPER = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
PIPE = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'pipeline_output' / 'classification'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return (list(reader.fieldnames or []), list(reader))

def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    enriched = 0
    _, ranking = read(PIPE / 'classification_shap_feature_ranking.csv')
    _, shap = read(PIPE / 'classification_shap_values.csv')
    ranks = {}
    for row in ranking:
        ranks.setdefault(row['target'], []).append(row)
    for target in ranks:
        ranks[target].sort(key=lambda row: float(row['rank']))
    lookup = {(row['target'], row['sample_index_task']): row for row in shap}
    for path in (PIPE / 'shap').rglob('classification_shap_dependence_*_plot_data.csv'):
        _, rows = read(path)
        target = rows[0]['target']
        feature = rows[0]['feature']
        color = ranks[target][1]['feature'] if feature == ranks[target][0]['feature'] else ranks[target][0]['feature']
        for row in rows:
            row['color_feature'] = color
            row['color_normalized_descriptor_value'] = lookup[target, row['sample_index_task']]['value_' + color]
            row['x_value'] = row['normalized_feature_value']
            row['y_value'] = row['shap_value']
            row['color_value'] = row['color_normalized_descriptor_value']
        write(path, rows)
        enriched += 1
    for path in (PAPER / 'dependence_plots').rglob('*_plot_data.csv'):
        _, rows = read(path)
        original = 'original_values' in path.parts
        shap_plot = 'SHAP_dependence' in path.parts
        for row in rows:
            row['x_value'] = row['raw_feature_value' if original else 'normalized_feature_value']
            row['y_value'] = row['shap_value'] if shap_plot else row['raw_target_value' if original else 'normalized_target_value']
            row['color_value'] = row['raw_target_value' if original else 'normalized_target_value'] if shap_plot else ''
            row['x_variable'] = row['feature']
            row['y_variable'] = 'SHAP value' if shap_plot else row['target']
            row['color_variable'] = row['target'] if shap_plot else ''
            row['display_scale'] = 'original' if original else 'normalized'
            for key in ['x_value', 'y_value'] + (['color_value'] if shap_plot else []):
                assert math.isfinite(float(row[key])), (path, key)
        write(path, rows)
        enriched += 1
    _, best = read(PIPE / 'best_classification_model_summary.csv')
    for row in best:
        path = PAPER / row['target'] / 'model_validation' / 'performance_validation_task_B_high_vs_low_plot_data.csv'
        _, rows = read(path)
        for sample in rows:
            for metric in ['BalancedAccuracy', 'F1', 'ROC_AUC', 'PR_AUC', 'MCC']:
                for stat in ['mean', 'std']:
                    sample['CV_' + metric + '_' + stat] = row[metric + '_' + stat]
        write(path, rows)
        enriched += 1
    figures = sorted({p.with_suffix('') for p in PAPER.rglob('*') if p.suffix.lower() in {'.png', '.pdf', '.svg'}})
    index, issues = ([], [])
    for stem in figures:
        path = stem.with_name(stem.name + '_plot_data.csv')
        fields, rows = read(path) if path.exists() else ([], [])
        if not rows:
            issues.append(str(path))
        index.append({'figure_stem': str(stem.relative_to(PAPER)).replace('\\', '/'), 'plot_data_csv': str(path.relative_to(PAPER)).replace('\\', '/'), 'plot_data_exists': path.exists(), 'rows': len(rows), 'columns': len(fields), 'column_names': ';'.join(fields), 'explicit_xyz_columns': all((key in fields for key in ['x_value', 'y_value', 'color_value']))})
    write(PAPER / 'all_figure_plot_data_index.csv', index)
    write(PAPER / 'all_figure_plot_data_audit.csv', [{'check': 'unique_figure_stems', 'count': len(figures)}, {'check': 'missing_or_empty_plot_data', 'count': len(issues)}, {'check': 'enriched_coordinate_or_annotation_tables', 'count': enriched}])
    assert not issues, issues
    print(f'Verified {len(figures)} figures: all CSVs readable and nonempty. Enriched {enriched} tables.')
if __name__ == '__main__':
    main()
