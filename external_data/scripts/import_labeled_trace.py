#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "external_data" / "labeled_events"
REQUIRED_COLUMNS = {"time", "lat", "lon", "speed", "heading"}


def main():
    parser = argparse.ArgumentParser(description="Import a real labeled trajectory event into SwarmMind.")
    parser.add_argument("csv", help="CSV with at least time, lat, lon, speed, heading columns")
    parser.add_argument("--label", choices=["normal", "wrongway", "construction", "fault", "drunk"], required=True)
    parser.add_argument("--name", required=True, help="Stable event name, e.g. dot_case_001")
    parser.add_argument("--source", default="restricted_partner_trace", help="Source label for provenance")
    args = parser.parse_args()

    src = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(src)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if "hr" not in df.columns:
        df["hr"] = 72.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{args.name}.csv"
    df[["time", "lat", "lon", "speed", "heading", "hr"]].to_csv(target, index=False)

    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"events": []}
    manifest["events"] = [event for event in manifest["events"] if event["name"] != args.name]
    manifest["events"].append({
        "name": args.name,
        "label": args.label,
        "source": args.source,
        "path": str(target.relative_to(ROOT)),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Imported {target}")


if __name__ == "__main__":
    main()
