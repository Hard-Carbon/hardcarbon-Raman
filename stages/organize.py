import csv
import hashlib
import json
from pathlib import Path
import shutil
PAPER = Path(__file__).resolve().parents[1] / 'result' / 'paper_results'
DEST = Path(__file__).resolve().parents[1] / 'result' / 'paper_results' / 'dependence_plots'
TARGETS = ['0.2C-R', 'Dp-2', 'Ds-2', 'SL-R']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    moves = []
    for target in TARGETS:
        source_root = PAPER / target / 'dependence'
        for scope in ['full99', 'test_oof50']:
            for scale in ['normalized', 'original_values']:
                source = source_root / (scope if scale == 'normalized' else scope + '_original_values')
                if not source.exists():
                    continue
                for item in source.iterdir():
                    if not item.is_file():
                        raise RuntimeError(f'Unexpected subdirectory: {item}')
                    if item.name.startswith('shap_dependence_'):
                        category = 'SHAP_dependence'
                    elif item.name.startswith('feature_target_'):
                        category = 'feature_vs_target'
                    else:
                        category = 'source_data'
                    destination = DEST / target / scope / category / scale / item.name
                    destination.resolve().relative_to(PAPER)
                    item.resolve().relative_to(PAPER)
                    if destination.exists():
                        raise RuntimeError(f'Destination already exists: {destination}')
                    moves.append((item, destination, digest(item)))
    if not moves:
        raise RuntimeError('No unorganized dependence files found.')
    mapping = {}
    for old, new, expected_hash in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        if digest(new) != expected_hash:
            raise RuntimeError(f'File changed during move: {new}')
        mapping[str(old.relative_to(PAPER)).replace('\\', '/')] = str(new.relative_to(PAPER)).replace('\\', '/')
    for target in TARGETS:
        source = PAPER / target / 'dependence'
        for directory in sorted(source.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and (not any(directory.iterdir())):
                directory.rmdir()
        if source.exists() and (not any(source.iterdir())):
            source.rmdir()
    for index_path in PAPER.glob('*index*.csv'):
        with index_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        changed = False
        for row in rows:
            for field in ['figure_stem', 'plot_data_csv', 'source_files']:
                if field not in row:
                    continue
                value = row[field]
                if field == 'figure_stem':
                    for extension in ['.png', '.pdf', '.svg']:
                        if value + extension in mapping:
                            row[field] = mapping[value + extension][:-len(extension)]
                            changed = True
                            break
                elif value in mapping:
                    row[field] = mapping[value]
                    changed = True
                elif field == 'source_files':
                    for old, new in mapping.items():
                        if old in row[field]:
                            row[field] = row[field].replace(old, new)
                            changed = True
        if changed:
            with index_path.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    rows = []
    for stem in sorted({p.with_suffix('') for p in PAPER.rglob('*') if p.suffix.lower() in {'.png', '.pdf', '.svg'}}):
        data_path = stem.with_name(stem.name + '_plot_data.csv')
        rows.append({'figure_stem': str(stem.relative_to(PAPER)).replace('\\', '/'), 'plot_data_csv': str(data_path.relative_to(PAPER)).replace('\\', '/'), 'plot_data_exists': data_path.exists()})
    with (PAPER / 'all_figure_plot_data_index.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing = [row for row in rows if not row['plot_data_exists']]
    with (PAPER / 'all_figure_plot_data_audit.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['check', 'count'])
        writer.writeheader()
        writer.writerows([{'check': 'unique_figure_stems', 'count': len(rows)}, {'check': 'missing_adjacent_plot_data', 'count': len(missing)}])
    with (DEST / 'file_move_manifest.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['previous_path', 'organized_path', 'sha256'])
        writer.writeheader()
        writer.writerows(({'previous_path': str(old.relative_to(PAPER)), 'organized_path': str(new.relative_to(PAPER)), 'sha256': sha} for old, new, sha in moves))
    comparison = []
    for target in TARGETS:
        for scope in ['full99', 'test_oof50']:
            for kind in ['SHAP_dependence', 'feature_vs_target']:
                normalized = DEST / target / scope / kind / 'normalized'
                original = DEST / target / scope / kind / 'original_values'
                norm_stems = {p.stem for p in normalized.glob('*.png')}
                raw_stems = {p.stem for p in original.glob('*.png')}
                assert norm_stems == raw_stems and len(norm_stems) == 8
                for stem in sorted(norm_stems):
                    comparison.append({'target': target, 'scope': scope, 'plot_type': kind, 'figure': stem, 'normalized_png': str((normalized / (stem + '.png')).relative_to(DEST)), 'original_png': str((original / (stem + '.png')).relative_to(DEST)), 'normalized_csv': str((normalized / (stem + '_plot_data.csv')).relative_to(DEST)), 'original_csv': str((original / (stem + '_plot_data.csv')).relative_to(DEST))})
    with (DEST / 'paired_plot_index.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    assert not missing, missing
    print(json.dumps({'moved_files': len(moves), 'paired_figures': len(comparison), 'all_figure_stems': len(rows), 'missing_plot_data': len(missing)}))
if __name__ == '__main__':
    main()
