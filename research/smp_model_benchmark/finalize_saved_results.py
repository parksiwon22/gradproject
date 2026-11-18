"""Regenerate summaries, figures and report from completed benchmark CSVs."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import benchmark as b


def main() -> None:
    daily = pd.read_csv(b.RESULTS / "daily_scores.csv")
    hourly = pd.read_csv(b.RESULTS / "hourly_predictions.csv")
    runtime = pd.read_csv(b.RESULTS / "model_runtime.csv")
    calibration = pd.read_csv(b.RESULTS / "rnn_sequence_calibration.csv")
    environment = json.loads((b.RESULTS / "environment.json").read_text(encoding="utf-8"))

    for index, row in runtime.iterrows():
        if row.model in ("LSTM", "GRU"):
            runtime.loc[index, "tree_node_or_linear_coefficient_count"] = None
            continue
        model = joblib.load(b.MODELS / f"{row.fold}_{str(row.model).lower()}_d3.joblib")
        params, complexity = b._model_complexity(model)
        runtime.loc[index, "parameter_count"] = params
        runtime.loc[index, "tree_node_or_linear_coefficient_count"] = complexity

    summary, seed_summary, hourly_mae = b._summarize(daily, hourly)
    bootstrap = b._paired_bootstrap(daily)
    summary.to_csv(b.RESULTS / "model_comparison.csv", index=False)
    seed_summary.to_csv(b.RESULTS / "neural_seed_variability.csv", index=False)
    hourly_mae.to_csv(b.RESULTS / "hourly_mae.csv", index=False)
    runtime.to_csv(b.RESULTS / "model_runtime.csv", index=False)
    bootstrap.to_csv(b.RESULTS / "paired_bootstrap_vs_hgb.csv", index=False)
    b._make_figures(summary, daily, hourly)
    b._write_report(summary, seed_summary, calibration, runtime, bootstrap, environment)

    manifest_path = b.RESULTS / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["future_leak_mutation_tests"] = 9
    manifest["metrics"] = "hourly pooled MAE/RMSE; daily mean EV regret; see daily_scores.csv for paired date values"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
