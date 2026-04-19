#!/usr/bin/env python3
"""
Fetch hotspot geometry from Overpass API and write GeoJSON-like JSON blobs.
This script is intentionally lightweight and only depends on the stdlib.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parents[1] / "osm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

QUERIES = {
    "anna_salai": """
        [out:json][timeout:25];
        way["highway"](around:1200,13.0490,80.2518);
        out geom;
    """,
    "i405": """
        [out:json][timeout:25];
        way["highway"](around:1600,33.9206,-118.3462);
        out geom;
    """,
    "us101_ngsim": """
        [out:json][timeout:25];
        way["highway"](around:1600,34.141354,-118.352898);
        out geom;
    """,
    "i95": """
        [out:json][timeout:25];
        way["highway"](around:1600,38.8051,-77.0468);
        out geom;
    """,
}


def main():
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    for name, query in QUERIES.items():
        payload = None
        last_error = None
        for attempt in range(3):
            try:
                data = urllib.parse.urlencode({"data": query}).encode("utf-8")
                req = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
                with urllib.request.urlopen(req, timeout=60, context=context) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2 + attempt)
        if payload is None:
            print(f"failed {name}: {last_error}")
            continue
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
