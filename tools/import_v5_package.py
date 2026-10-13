"""Materialize only private research inputs from an already accessible V5 ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def import_package(package: Path):
    manifest = json.loads((ROOT / 'benchmarks/v5/source_manifest.json').read_text())
    entries = {e['zip_path']: e for e in manifest}
    selected = {'pkg/01_model_v5/data/aligned_hourly_all_features_2023.csv': ROOT / 'private_data/aligned_hourly_all_features_2023.csv',
                'pkg/01_model_v5/results/v5_day_ahead_predictions.csv': ROOT / 'private_reference/v5_day_ahead_predictions.csv'}
    with zipfile.ZipFile(package) as archive:
        contents = {}
        for name, destination in selected.items():
            entry = entries[name]
            if archive.getinfo(name).file_size != entry['bytes']:
                raise ValueError('V5 input size mismatch: ' + name)
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('V5 input hash mismatch: ' + name)
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError('Refusing to overwrite different private data: ' + str(destination))
            contents[destination] = data
        for destination, data in contents.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    return list(selected.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path)
    args = parser.parse_args()
    package = args.package
    if package is None:
        candidates = sorted(ROOT.parent.glob('project_sources/*smp_project_v5_package.zip'))
        if len(candidates) != 1:
            parser.error('Exactly one accessible original V5 ZIP is required; use --package. This tool cannot access unseen ChatGPT attachments.')
        package = candidates[0]
    for path in import_package(package):
        print(path.relative_to(ROOT))

if __name__ == '__main__':
    main()
