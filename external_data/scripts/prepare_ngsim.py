#!/usr/bin/env python3
"""
Prepare FHWA NGSIM trajectory CSVs into a SwarmMind-friendly schema.

Expected raw columns vary by source package, so this script is deliberately
conservative: it normalizes common NGSIM column names when they are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "time": ["Global_Time", "global_time", "time"],
    "vehicle_id": ["Vehicle_ID", "vehicle_id", "id"],
    "speed": ["v_Vel", "speed"],
    "lane": ["Lane_ID", "lane_id", "lane"],
    "x": ["Local_X", "x"],
    "y": ["Local_Y", "y"],
}


def pick_column(df: pd.DataFrame, options: list[str]) -> str | None:
    for name in options:
        if name in df.columns:
            return name
    return None


def normalize_ngsim(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for target, aliases in COLUMN_ALIASES.items():
        chosen = pick_column(df, aliases)
        if chosen is not None:
            out[target] = df[chosen]
    norm = pd.DataFrame(out)
    if "time" in norm.columns:
        norm["time"] = (norm["time"] - norm["time"].min()) / 1000.0
    return norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    norm = normalize_ngsim(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
