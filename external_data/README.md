# External Data Plan

This folder is the staging area for the real-world evidence SwarmMind needs to feel validated, credible, and demo-proof.

## Current Workspace Status
- `NGSIM`: staged in `external_data/ngsim/ngsim_full.csv`
- `GeoLife raw GPS`: adapter-ready; download from Microsoft and stage with `scripts/prepare_geolife.py`
- `OSM/Overpass`: staged for `Anna Salai`, `I-405`, `I-95`, and `US-101 NGSIM`
- `OpenCelliD`: still pending because the public dataset/API requires a registered key
- `WZDx`: integration-ready; add a direct WZDx GeoJSON feed with `scripts/fetch_wzdx_feed.py`
- `Reports`: cataloged and ready to cite in slides/README

## Priority Order
1. `NGSIM` trajectories for real-world validation.
2. `OSM / Overpass` geometry for real map context.
3. `OpenCelliD` tower density for Layer 9 realism.
4. `NTSB / NHTSA` safety reports for motivation.
5. `rPPG bias / privacy` references for ethics.
6. `demo_video/` backup assets for judging resilience.

## Suggested Layout
```text
external_data/
  catalog.json
  README.md
  scripts/
    fetch_osm_hotspots.py
    fetch_opencellid.py
    prepare_ngsim.py
    extract_ngsim_demo_traces.py
    summarize_catalog.py
  ngsim/
    us101/
    i80/
  geolife/
  labeled_events/
  osm/
    anna_salai.geojson
    i405.geojson
    i95.geojson
  opencellid/
  wzdx/
  reports/
  demo_video/
```

## Judge-Facing Outcomes
- "Validated on FHWA NGSIM real-world freeway trajectories plus synthetic stress cases."
- "Compared against a simple heading-threshold baseline."
- "Grounded in real road geometry and real connectivity density."
- "Protected by an offline backup demo."
- "Explicitly documents rPPG privacy and bias limitations."

## Connectivity and Work-Zone Reality Checks
`OpenCelliD` is intentionally not faked. Until a user-owned key or downloaded extract is placed in `external_data/opencellid/`, Layer 9 reports synthetic tower context in the app evidence panel.

`WZDx` feeds can be added without code changes:

```bash
python3 external_data/scripts/fetch_wzdx_feed.py "https://example.com/wzdx.geojson" --name us101
```

When `external_data/wzdx/us101.geojson` exists, the app overlays those work-zone lines and the evidence panel switches from `synthetic/no feed` to `WZDx feed`.

## Raw GPS Reality Check
NGSIM is real trajectory data, but it is not raw GPS. To make a separate raw-GPS claim, stage Microsoft GeoLife:

```bash
python3 external_data/scripts/prepare_geolife.py "/path/to/Geolife Trajectories 1.3.zip"
```

After `.plt` files exist under `external_data/geolife/`, the app evidence panel reports `Microsoft GeoLife` as the raw GPS source. GeoLife does not contain labeled wrong-way crashes; it is used as raw-GPS mobility validation and negative calibration data.

## Real Labeled Wrong-Way Events
Public real wrong-way crash trajectories are not safely or reliably available in this repo. If a DOT, OEM, simulator operator, or research partner provides a permitted real labeled CSV, import it with:

```bash
python3 external_data/scripts/import_labeled_trace.py "/path/to/case.csv" --label wrongway --name dot_case_001 --source partner_dot
```

Required columns:

```text
time,lat,lon,speed,heading
```

Optional column:

```text
hr
```

This changes the evidence panel from `real labeled events: not staged` to `staged` and upgrades the report validation tier when the other external feeds are present.
