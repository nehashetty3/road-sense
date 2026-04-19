#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real_data import load_real_ngsim_trace


OUT_DIR = Path(__file__).resolve().parents[1] / "ngsim" / "prepared"


def main():
    out_dir = OUT_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out_dir = Path("/tmp/swarmmind_ngsim_prepared")
        out_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "us101_real_normal.csv": "normal",
        "us101_real_wrongway.csv": "wrongway_injected",
        "us101_real_construction.csv": "construction_injected",
    }
    for filename, variant in variants.items():
        df = load_real_ngsim_trace("us101", variant)
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
