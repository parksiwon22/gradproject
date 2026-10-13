"""Run the portable V5 baseline without overwriting archived benchmarks."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--full', action='store_true')
    parser.add_argument('--data', type=Path, default=ROOT / 'private_data/aligned_hourly_all_features_2023.csv')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/v5')
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('Private V5 data unavailable. Run tools/import_v5_package.py with an accessible original ZIP; never substitute synthetic data for benchmark results.')
    run_dir = args.run_dir.resolve()
    protected = [ROOT / 'benchmarks', ROOT / 'model_v5/reference', ROOT / 'outputs']
    if any(run_dir == p or p in run_dir.parents for p in protected):
        parser.error('Run directory must not overwrite preserved references or web outputs.')
    from model_v5 import smp_predictor_v5 as model
    model.DATA_PATH = args.data.resolve()
    model.RESULTS_DIR = run_dir / 'results'
    model.MODELS_DIR = run_dir / 'models'
    sys.argv = [sys.argv[0], '--quick' if args.quick else '--full']
    model.main()

if __name__ == '__main__':
    main()
