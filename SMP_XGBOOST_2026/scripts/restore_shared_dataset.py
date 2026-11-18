"""Expand the bundled common dataset into the layout expected by the trainer."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "data" / "shared_dataset_v5"
MANIFEST = SOURCE_DIR / "manifest.json"


def main() -> None:
    dataset_version = json.loads(MANIFEST.read_text(encoding="utf-8"))["dataset_version"]
    target_dir = PROJECT_DIR.parent / "SMP_SHARED_DATA" / "versions" / dataset_version
    if target_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing dataset directory: {target_dir}")

    target_dir.mkdir(parents=True)
    for source in sorted(SOURCE_DIR.iterdir()):
        if source.suffix == ".gz":
            target = target_dir / source.stem
            with gzip.open(source, "rb") as compressed, target.open("wb") as restored:
                shutil.copyfileobj(compressed, restored, 1024 * 1024)
        elif source.is_file():
            shutil.copy2(source, target_dir / source.name)

    print(f"Restored {dataset_version} to {target_dir}")


if __name__ == "__main__":
    main()
