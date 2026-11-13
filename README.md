# Heptagonal Topological Defects Determine Sodium Storage Kinetics in Hard Carbon

This repository contains the Raman-based classification code and processed input data accompanying the study.

## Requirements

Install the Python dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Repository Structure

| Path | Contents |
| --- | --- |
| `data/` | Input data |
| `classification.py` | Classification entry point |
| `config.yaml` | Workflow settings |
| `src/` | Data loading, models, and visualization |
| `main.py`, `stages/`, `run.bat` | Scripts for the full analysis workflow |

## Demo

The four CSV files contain seven Raman descriptors and one electrochemical target per file. The classification workflow compares the high and low groups defined by the inclusive 75th and 25th percentiles, respectively.

## Usage

Set `normalized_data_path` in `config.yaml` to the CSV paths under `data/`, then run the classification script from the repository root.

```bash
python classification.py --config config.yaml
```

Classification outputs are written under the configured `output_dir`. The full workflow in `main.py` requires additional raw and reference inputs.

## License

The code is available under the [MIT License](LICENSE).
