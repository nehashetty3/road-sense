#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parents[1] / "opencellid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://opencellid.org/cell/getInArea"

BBOXES = {
    "anna_salai": "13.000,80.200,13.090,80.310",
    "i405": "33.880,-118.410,33.970,-118.280",
    "i95": "38.760,-77.120,38.870,-76.930",
    "us101_ngsim": "34.100,-118.410,34.180,-118.300",
}


def main():
    api_key = os.environ.get("OPENCELLID_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENCELLID_API_KEY. Register for an OpenCelliD key, then run "
            "`OPENCELLID_API_KEY=... python3 external_data/scripts/fetch_opencellid.py`."
        )

    for name, bbox in BBOXES.items():
        params = {
            "key": api_key,
            "BBOX": bbox,
            "format": "json",
            "limit": 1000,
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
