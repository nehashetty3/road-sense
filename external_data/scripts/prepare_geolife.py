#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "external_data" / "geolife"


def main():
    parser = argparse.ArgumentParser(description="Stage Microsoft GeoLife raw GPS .plt files for SwarmMind.")
    parser.add_argument("source", help="Path to a GeoLife zip file or extracted GeoLife folder")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            zf.extractall(OUT_DIR)
    elif source.is_dir():
        for plt in source.rglob("*.plt"):
            rel = plt.relative_to(source)
            target = OUT_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plt, target)
    else:
        raise SystemExit(f"Unsupported GeoLife source: {source}")

    count = len(list(OUT_DIR.rglob("*.plt")))
    print(f"Staged {count} GeoLife .plt files under {OUT_DIR}")


if __name__ == "__main__":
    main()
