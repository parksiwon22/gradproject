"""Compare local V5 reproduction with a privately held original ZIP."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from model.v5 import DATA_PATH, RESULTS_DIR


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.zip) as archive:
        prefix = "pkg/01_model_v5/"
        original = pd.read_csv(BytesIO(archive.read(prefix + "results/v5_model_comparison.csv")))
        reproduced = pd.read_csv(RESULTS_DIR / "v5_model_comparison.csv")
        old_predictions = pd.read_csv(BytesIO(archive.read(prefix + "results/v5_day_ahead_predictions.csv")))
        new_predictions = pd.read_csv(RESULTS_DIR / "v5_day_ahead_predictions.csv")
        selected = "validation_selected_profile_hgb_blend"
        row = reproduced[(reproduced.split == "final_test") & (reproduced.model == selected)].iloc[0]
        original_row = original[(original.split == "final_test") & (original.model == selected)].iloc[0]
        report = {
            "python": platform.python_version(),
            "platform": platform.system(),
            "dependencies": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib", "torch", "streamlit"]},
            "zip_sha256": digest(args.zip.read_bytes()),
            "data_sha256": digest(DATA_PATH.read_bytes()),
            "zip_data_sha256": digest(archive.read(prefix + "data/aligned_hourly_all_features_2023.csv")),
            "original_source_sha256": digest(archive.read(prefix + "smp_predictor_v5.py")),
            "app_text_identical": archive.read("pkg/02_web/app.py").decode("utf-8").splitlines() == Path("app.py").read_text(encoding="utf-8").splitlines(),
            "MAE": float(row.MAE), "R2": float(row.R2),
            "MAE_delta": float(row.MAE - original_row.MAE),
            "R2_delta": float(row.R2 - original_row.R2),
            "max_prediction_delta": float(np.max(np.abs(old_predictions.profile_hgb_blend - new_predictions.profile_hgb_blend))),
            "zip_inventory": archive.namelist(),
            "repository_original_inventory": subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "b2d4f64"], text=True).splitlines(),
        }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/reproduction.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["python", "MAE", "R2", "MAE_delta", "R2_delta", "max_prediction_delta", "app_text_identical"]}, indent=2))
    if report["data_sha256"] != report["zip_data_sha256"] or report["max_prediction_delta"] > 1e-8:
        raise AssertionError("Reproduction differs from original ZIP")


if __name__ == "__main__":
    main()
