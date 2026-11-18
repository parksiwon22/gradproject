# LightGBM 72-hour SMP upgrade

This folder contains a reproducible upgrade experiment for the existing 72-hour LightGBM land-SMP model. The existing project, model, input bundle, and prior results remain unchanged.

## Run

From the repository root, install the dependencies from `SMP_LightGBM_2026/requirements.txt`, then run:

```powershell
python SMP_LightGBM_2026/upgrades/upgrade_72h_20261119/scripts/run_lightgbm_upgrade.py
python SMP_LightGBM_2026/upgrades/upgrade_72h_20261119/scripts/evaluate_lightgbm_upgrade.py
```

The training script discovers the frozen common v5 dataset by its manifest. To select it explicitly, pass `--dataset-root <folder>` or set `SMP72H_DATASET_ROOT`. The dataset is read in place and is not duplicated in this upgrade folder.

The existing LightGBM v4 run is the E0 reproduction reference. Validation selects blending weights and the candidate configuration. The validation-selected final rule is E1 (existing LightGBM plus the weekly baseline at α=0.8); the Train+Validation refit is preserved separately as an evaluated comparison. Both are scored on the same Final Test rows, and no candidate was switched after reading Final Test results. The report identifies the reused Final Test period and the unresolved conditional data scope.

## Contents

- `PROMPT_LIGHTGBM.md`: the provided work prompt adapted for LightGBM and its output paths.
- `config/experiment_config.json`: selected hyperparameters, iteration count, data versions, and blend weights.
- `models/final_lightgbm_model.txt`: model component used by the selected E1 blend.
- `models/train_validation_refit_lightgbm_model.txt`: newly trained E2 model using Train+Validation.
- `outputs/predictions/`: Validation experiment rows and Final Test predictions.
- `outputs/metrics/`: comparison tables, horizon/day/month metrics, bias, source-row reproduction, coverage, and artifact verification.
- `outputs/figures/`: model comparison, one 72-hour example, monthly MAE, and horizon MAE.
- `reports/`: baseline, retraining, bias, data gate, and final reports.

Prediction rows carry both `dataset_version` (the LightGBM run alias) and `source_dataset_version` (the exact immutable manifest ID), so the data provenance remains traceable while the model output is named consistently.
