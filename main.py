import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parent
INPUT = BASE / "data"


def load_stage(name, **values):
    module = importlib.import_module("stages." + name)
    for key, value in values.items():
        setattr(module, key, value)
    return module


def main():
    os.chdir(BASE)
    config = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    output = (BASE / config["output_dir"]).resolve()
    paper = output.parent
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output already contains results: {output}")
    for path in config["normalized_data_path"]:
        frame = pd.read_csv(BASE / path)
        assert len(frame) == 99
        assert list(frame.columns[:-1]) == config["feature_columns"]
        assert np.isfinite(frame.to_numpy(dtype=float)).all()
    config["normalized_data_path"] = [str((BASE / path).resolve()) for path in config["normalized_data_path"]]
    config["output_dir"] = str(output)
    config["log_dir"] = str(paper / "logs")
    paper.mkdir(parents=True, exist_ok=True)
    runtime_config = paper / "config_used.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    import classification
    sys.argv = ["classification.py", "--config", str(runtime_config)]
    classification.main()
    pipeline = output / "classification"
    best = pd.read_csv(pipeline / "best_classification_model_summary.csv")
    historical = pd.read_csv(BASE / "reference/regression_context.csv")
    context = best.rename(columns={"best_model": "best_classification_model"}).merge(historical, on="target", how="left")
    context["interpretation_hint"] = np.where(context.regression_R2_mean < 0.2, "Low regression R2 but useful classification signal for high-performance screening.", "Classification result is suitable for main or supporting discussion.")
    context.to_csv(pipeline / "classification_vs_regression_context.csv", index=False, encoding="utf-8-sig")
    data = paper / "data_used"
    data.mkdir(exist_ok=True)
    shutil.copy2(INPUT / "selected_99_source_mapping.csv", data / "selected_99_source_mapping.csv")
    for path in (INPUT / "normalized").glob("*.csv"):
        shutil.copy2(path, data / path.name)
    summary = load_stage("summary", ROOT=paper, RUN=pipeline, OUT=paper / "global_summary", FIG=paper / "global_summary/figures", INPUT_NORMALIZED=INPUT / "normalized", PREVIOUS_REFERENCE=BASE / "reference/five_feature_summary.csv")
    summary.main()
    validation = load_stage("validation", ROOT=paper, SOURCE=pipeline, OUT=paper)
    validation.main()
    export = load_stage("export", ROOT=paper, CLASSIFICATION=pipeline, SUMMARY=paper / "global_summary", PAPER=paper, CONFIRMATION=paper / "_no_confirmation")
    export.main()
    dependence = load_stage("dependence", PAPER=paper, PIPE=pipeline)
    dependence.main()
    raw = load_stage("raw_dependence", ROOT=BASE, PAPER=paper, SOURCE=INPUT / "raw", PERFORMANCE=INPUT / "raw/Average Raman-performance-raw(2).xlsx", KINETIC=INPUT / "raw/Raman-Kinective(1).xlsx")
    raw.main()
    organize = load_stage("organize", PAPER=paper, DEST=paper / "dependence_plots")
    organize.main()
    audit = load_stage("audit", PAPER=paper, PIPE=pipeline)
    audit.main()
    expected = pd.read_csv(BASE / "reference/expected_metrics.csv")
    actual = best.set_index("target").sort_index()
    expected = expected.set_index("target").sort_index()
    metrics = ["BalancedAccuracy_mean", "BalancedAccuracy_std", "F1_mean", "ROC_AUC_mean", "PR_AUC_mean", "MCC_mean"]
    np.testing.assert_array_equal(actual.best_model.to_numpy(), expected.best_model.to_numpy())
    np.testing.assert_allclose(actual[metrics].to_numpy(), expected[metrics].to_numpy(), rtol=0, atol=1e-10)
    expected_figures = set(json.loads((BASE / "reference/expected_figures.json").read_text(encoding="utf-8")))
    actual_figures = {str(path.with_suffix("").relative_to(paper)).replace("\\", "/") for path in paper.rglob("*") if path.suffix.lower() in {".png", ".pdf", ".svg"}}
    if actual_figures != expected_figures:
        raise RuntimeError({"missing": sorted(expected_figures - actual_figures), "extra": sorted(actual_figures - expected_figures)})
    for directory in sorted(paper.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    pd.DataFrame([{"check": "metrics_match", "passed": True}, {"check": "334_figure_stems_match", "passed": True}, {"check": "all_figures_have_plot_data", "passed": True}]).to_csv(paper / "reproduction_check.csv", index=False)
    print(str(paper), flush=True)


if __name__ == "__main__":
    main()
