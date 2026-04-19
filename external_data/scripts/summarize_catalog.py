#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog.json"


def main():
    payload = json.loads(CATALOG.read_text())
    print("SwarmMind external data status\n")
    for item in payload.get("datasets", []):
        print(f"- {item['title']} [{item['status']}]")
        print(f"  purpose: {item['purpose']}")
        print(f"  local:   {item['local_path']}")
        print(f"  source:  {item['source_url']}")


if __name__ == "__main__":
    main()
