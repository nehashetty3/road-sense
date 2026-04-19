from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "external_data" / "wzdx"
REGISTRY_URL = "https://data.transportation.gov/api/views/69qe-yiui/rows.json?accessType=DOWNLOAD"


def main():
    parser = argparse.ArgumentParser(description="Fetch a WZDx GeoJSON feed for SwarmMind.")
    parser.add_argument("url", nargs="?", help="Direct WZDx GeoJSON feed URL")
    parser.add_argument("--name", default="us101", help="Output file stem")
    parser.add_argument("--registry", action="store_true", help="Save the USDOT WZDx feed registry metadata instead of a feed")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.registry:
        with urlopen(REGISTRY_URL) as response:
            payload = json.load(response)
        out_path = OUT_DIR / "feed_registry.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out_path}")
        return

    if not args.url:
        raise SystemExit("Pass a direct WZDx GeoJSON feed URL, or use --registry.")

    with urlopen(args.url) as response:
        payload = json.load(response)

    out_path = OUT_DIR / f"{args.name}.geojson"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
