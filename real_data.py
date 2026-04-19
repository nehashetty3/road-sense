from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXTERNAL_DIR = ROOT / "external_data"
NGSIM_PATH = EXTERNAL_DIR / "ngsim" / "ngsim_full.csv"
OSM_DIR = EXTERNAL_DIR / "osm"
OPENCELLID_DIR = EXTERNAL_DIR / "opencellid"
WZDX_DIR = EXTERNAL_DIR / "wzdx"
GEOLIFE_DIR = EXTERNAL_DIR / "geolife"
LABELED_EVENTS_DIR = EXTERNAL_DIR / "labeled_events"
CURATED_DIR = EXTERNAL_DIR / "ngsim" / "curated"
OSM_HOTSPOT_FILES = {
    "anna_salai": "anna_salai.json",
    "i405": "i405.json",
    "i95": "i95.json",
    "us101": "us101_ngsim.json",
}
_NGSIM_DF_CACHE = None

REAL_LOCATIONS = {
    "us101": {
        "label": "US-101 NGSIM, Los Angeles",
        "center": (34.141354, -118.352898),
        "road_direction": 130.0,
        "osm_file": "us101_ngsim.json",
        "ngsim_location": "us-101",
    },
    "i80": {
        "label": "I-80 NGSIM, Emeryville",
        "center": (37.8401, -122.2913),
        "road_direction": 90.0,
        "osm_file": None,
        "ngsim_location": "i-80",
    },
}


def has_real_ngsim_data() -> bool:
    return NGSIM_PATH.exists() and NGSIM_PATH.stat().st_size > 1_000_000


def has_real_workzone_data() -> bool:
    return WZDX_DIR.exists() and any(WZDX_DIR.glob("*.geojson"))


def has_raw_gps_data() -> bool:
    return GEOLIFE_DIR.exists() and any(GEOLIFE_DIR.rglob("*.plt"))


def has_real_labeled_events() -> bool:
    manifest_path = LABELED_EVENTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    return bool(manifest.get("events"))


def available_real_locations() -> dict:
    return {k: v for k, v in REAL_LOCATIONS.items() if has_real_ngsim_data()}


def _curated_trace_dir(location_key: str) -> Path:
    return CURATED_DIR / location_key


def _curated_manifest_path(location_key: str) -> Path:
    return _curated_trace_dir(location_key) / "manifest.json"


def load_overpass_features(key: str):
    filename = OSM_HOTSPOT_FILES.get(key)
    if not filename:
        return []
    path = OSM_DIR / filename
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    features = []
    for element in payload.get("elements", []):
        geom = element.get("geometry")
        if not geom or element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        features.append({
            "type": "Feature",
            "properties": {
                "highway": tags.get("highway", "road"),
                "name": tags.get("name", f"{key}-{element.get('id')}"),
                "oneway": tags.get("oneway") in {"yes", "1", "true"},
                "construction": tags.get("highway") == "construction" or "construction" in tags,
                "cycleway": tags.get("highway") == "cycleway" or "cycleway" in tags,
                "lanes": int(tags.get("lanes", 1)) if str(tags.get("lanes", "1")).isdigit() else 1,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[point["lon"], point["lat"]] for point in geom],
            },
        })
    return features


def load_real_towers(key: str):
    path = OPENCELLID_DIR / f"{key}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    towers = []
    for item in payload.get("cells", payload.get("results", [])):
        lat = item.get("lat") or item.get("latitude")
        lon = item.get("lon") or item.get("longitude")
        if lat is None or lon is None:
            continue
        towers.append({
            "lat": float(lat),
            "lon": float(lon),
            "signal": 0.75,
            "type": item.get("radio", item.get("type", "CELL")),
        })
    return towers


def load_workzone_events(key: str):
    path = WZDX_DIR / f"{key}.geojson"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("features", [])


def _parse_geolife_plt(path: Path, max_points: int = 600):
    rows = []
    with path.open() as handle:
        for line_no, line in enumerate(handle):
            if line_no < 6:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            rows.append({
                "lat": float(parts[0]),
                "lon": float(parts[1]),
                "alt": float(parts[3]),
                "date": parts[5],
                "clock": parts[6],
            })
            if len(rows) >= max_points:
                break
    if len(rows) < 5:
        raise ValueError(f"Not enough GPS points in {path}")
    df = pd.DataFrame(rows)
    df["time"] = np.arange(len(df), dtype=float)
    dlat = np.diff(df["lat"], prepend=df["lat"].iloc[0])
    dlon = np.diff(df["lon"], prepend=df["lon"].iloc[0])
    df["heading"] = (np.degrees(np.arctan2(dlon, dlat)) + 360.0) % 360.0
    df.loc[df.index[0], "heading"] = df["heading"].iloc[1]
    dist_m = np.sqrt((dlat * 111_000.0) ** 2 + (dlon * 111_000.0 * np.cos(np.radians(df["lat"]))) ** 2)
    df["speed"] = np.clip(dist_m * 3.6, 0.0, 130.0)
    df["hr"] = 72.0
    return df[["time", "lat", "lon", "speed", "heading", "hr"]]


def load_geolife_trace(trace_index: int = 0, max_points: int = 600):
    paths = sorted(GEOLIFE_DIR.rglob("*.plt"))
    if not paths:
        raise FileNotFoundError(f"Missing GeoLife .plt files under {GEOLIFE_DIR}")
    return _parse_geolife_plt(paths[min(trace_index, len(paths) - 1)], max_points=max_points)


def load_labeled_events():
    manifest_path = LABELED_EVENTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    events = []
    for item in manifest.get("events", []):
        path = ROOT / item["path"]
        if path.exists():
            event = dict(item)
            event["dataframe"] = pd.read_csv(path)
            events.append(event)
    return events


def _read_ngsim_columns():
    global _NGSIM_DF_CACHE
    if _NGSIM_DF_CACHE is not None:
        return _NGSIM_DF_CACHE.copy()
    usecols = [
        "Vehicle_ID", "Frame_ID", "Global_Time", "Global_X", "Global_Y",
        "v_Vel", "Lane_ID", "Location",
    ]
    _NGSIM_DF_CACHE = pd.read_csv(NGSIM_PATH, usecols=usecols)
    return _NGSIM_DF_CACHE.copy()


def _meters_to_latlon(dx_m: np.ndarray, dy_m: np.ndarray, anchor_lat: float, anchor_lon: float):
    lat = anchor_lat + dy_m / 111_000.0
    lon = anchor_lon + dx_m / (111_000.0 * np.cos(np.radians(anchor_lat)))
    return lat, lon


def _build_trace_from_vehicle(vehicle_df: pd.DataFrame, anchor_lat: float, anchor_lon: float):
    vehicle_df = vehicle_df.sort_values("Global_Time").copy()
    vehicle_df = vehicle_df.drop_duplicates(subset=["Global_Time"])
    # NGSIM is 10 Hz; downsample to 1 Hz for app parity.
    vehicle_df["sec_bucket"] = ((vehicle_df["Global_Time"] - vehicle_df["Global_Time"].min()) / 1000.0).round().astype(int)
    vehicle_df = vehicle_df.groupby("sec_bucket", as_index=False).first()

    x_ft = vehicle_df["Global_X"].to_numpy(dtype=float)
    y_ft = vehicle_df["Global_Y"].to_numpy(dtype=float)
    x_m = (x_ft - x_ft[0]) * 0.3048
    y_m = (y_ft - y_ft[0]) * 0.3048
    lat, lon = _meters_to_latlon(x_m, y_m, anchor_lat, anchor_lon)

    dx = np.diff(x_m, prepend=x_m[0])
    dy = np.diff(y_m, prepend=y_m[0])
    heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    heading[0] = heading[1] if len(heading) > 1 else 0.0

    speed_kmh = vehicle_df["v_Vel"].to_numpy(dtype=float) * 1.09728
    hr = np.full(len(vehicle_df), 72.0)

    return pd.DataFrame({
        "time": vehicle_df["sec_bucket"].astype(float),
        "lat": lat,
        "lon": lon,
        "speed": np.round(speed_kmh, 2),
        "heading": np.round(heading, 2),
        "hr": hr,
    })


def _candidate_vehicle_ids(location_key: str = "us101", min_seconds: int = 180):
    if not has_real_ngsim_data():
        raise FileNotFoundError(f"Missing NGSIM CSV at {NGSIM_PATH}")
    meta = REAL_LOCATIONS[location_key]
    df = _read_ngsim_columns()
    df = df[df["Location"].str.lower() == meta["ngsim_location"]]
    lengths = (
        df.groupby("Vehicle_ID")["Global_Time"]
        .nunique()
        .sort_values(ascending=False)
    )
    candidate_ids = lengths[lengths >= min_seconds * 10].index.tolist()
    if not candidate_ids:
        candidate_ids = lengths.index.tolist()
    return df, candidate_ids


def build_curated_real_trace_cache(
    location_key: str = "us101",
    max_traces: int = 4,
    min_seconds: int = 180,
    force: bool = False,
):
    trace_dir = _curated_trace_dir(location_key)
    manifest_path = _curated_manifest_path(location_key)
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    trace_dir.mkdir(parents=True, exist_ok=True)
    df, candidate_ids = _candidate_vehicle_ids(location_key, min_seconds=min_seconds)
    meta = REAL_LOCATIONS[location_key]
    traces = []
    for rank, vehicle_id in enumerate(candidate_ids[:max_traces], start=1):
        vehicle_df = df[df["Vehicle_ID"] == vehicle_id].copy()
        trace = _build_trace_from_vehicle(vehicle_df, meta["center"][0], meta["center"][1])
        trace_path = trace_dir / f"trace_{rank:02d}_vehicle_{vehicle_id}.csv"
        trace.to_csv(trace_path, index=False)
        traces.append({
            "rank": rank,
            "vehicle_id": int(vehicle_id),
            "seconds": int(trace["time"].max()) if len(trace) else 0,
            "path": str(trace_path.relative_to(ROOT)),
        })

    manifest = {
        "location_key": location_key,
        "label": meta["label"],
        "road_direction": meta["road_direction"],
        "count": len(traces),
        "traces": traces,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_curated_real_traces(location_key: str = "us101", max_traces: int = 4, min_seconds: int = 180):
    manifest = build_curated_real_trace_cache(location_key, max_traces=max_traces, min_seconds=min_seconds)
    traces = []
    for item in manifest.get("traces", []):
        path = ROOT / item["path"]
        if path.exists():
            traces.append(pd.read_csv(path))
    return traces


def load_real_ngsim_trace(
    location_key: str = "us101",
    variant: str = "normal",
    min_seconds: int = 180,
    trace_index: int = 0,
):
    traces = load_curated_real_traces(location_key, min_seconds=min_seconds)
    if not traces:
        raise FileNotFoundError(f"No curated real traces available for {location_key}")
    trace = traces[min(trace_index, len(traces) - 1)].copy()
    if variant == "normal":
        return trace
    if variant == "wrongway_injected":
        return inject_wrongway(trace)
    if variant == "short_wrongway_injected":
        return inject_short_wrongway(trace)
    if variant == "construction_injected":
        return inject_construction(trace)
    if variant == "lane_confused_injected":
        return inject_lane_confusion(trace)
    if variant == "stop_reverse_injected":
        return inject_stop_reverse(trace)
    raise ValueError(f"Unsupported variant: {variant}")


def inject_wrongway(df: pd.DataFrame, start_t: int = 60, duration: int = 45):
    out = df.copy()
    idx = (out["time"] >= start_t) & (out["time"] < start_t + duration)
    seg = out.loc[idx].copy()
    if len(seg) < 5:
        return out
    rev = seg.iloc[::-1].copy().reset_index(drop=True)
    rev["time"] = seg["time"].to_numpy()
    dx = np.diff(rev["lon"], prepend=rev["lon"].iloc[0])
    dy = np.diff(rev["lat"], prepend=rev["lat"].iloc[0])
    rev["heading"] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    rev["speed"] = np.clip(rev["speed"] * 0.85, 15.0, None)
    rev["hr"] = np.linspace(88, 122, len(rev))
    out.loc[idx, ["lat", "lon", "heading", "speed", "hr"]] = rev[["lat", "lon", "heading", "speed", "hr"]].to_numpy()
    return out


def inject_short_wrongway(df: pd.DataFrame, start_t: int = 60, duration: int = 70):
    return inject_wrongway(df, start_t=start_t, duration=duration)


def inject_construction(df: pd.DataFrame, start_t: int = 60, duration: int = 45):
    out = df.copy()
    idx = (out["time"] >= start_t) & (out["time"] < start_t + duration)
    if idx.sum() == 0:
        return out
    lat = out.loc[idx, "lat"].to_numpy()
    lon = out.loc[idx, "lon"].to_numpy()
    offset = np.linspace(0, 7, idx.sum()) / 111_000.0
    out.loc[idx, "lat"] = lat + offset
    out.loc[idx, "heading"] = (out.loc[idx, "heading"] + np.linspace(18, 42, idx.sum())) % 360.0
    out.loc[idx, "speed"] = np.clip(out.loc[idx, "speed"] * 0.7, 10.0, None)
    out.loc[idx, "hr"] = 76.0
    return out


def inject_lane_confusion(df: pd.DataFrame, start_t: int = 55, duration: int = 30):
    out = df.copy()
    idx = (out["time"] >= start_t) & (out["time"] < start_t + duration)
    if idx.sum() == 0:
        return out
    wave = np.sin(np.linspace(0, 2.5 * np.pi, idx.sum())) * (2.5 / 111_000.0)
    out.loc[idx, "lat"] = out.loc[idx, "lat"].to_numpy() + wave
    out.loc[idx, "heading"] = (out.loc[idx, "heading"] + np.sin(np.linspace(0, 2 * np.pi, idx.sum())) * 18.0) % 360.0
    out.loc[idx, "speed"] = np.clip(out.loc[idx, "speed"] * 0.82, 18.0, None)
    out.loc[idx, "hr"] = np.linspace(78.0, 90.0, idx.sum())
    return out


def inject_stop_reverse(df: pd.DataFrame, start_t: int = 70, stop_duration: int = 8, reverse_duration: int = 10):
    out = df.copy()
    stop_idx = (out["time"] >= start_t) & (out["time"] < start_t + stop_duration)
    reverse_idx = (out["time"] >= start_t + stop_duration) & (out["time"] < start_t + stop_duration + reverse_duration)
    if stop_idx.sum() == 0 or reverse_idx.sum() == 0:
        return out
    out.loc[stop_idx, "speed"] = np.linspace(12.0, 0.0, stop_idx.sum())
    out.loc[stop_idx, "hr"] = np.linspace(74.0, 80.0, stop_idx.sum())

    rev = out.loc[reverse_idx, ["lat", "lon", "heading", "speed", "hr"]].copy().iloc[::-1].reset_index(drop=True)
    rev["time"] = out.loc[reverse_idx, "time"].to_numpy()
    rev["speed"] = np.linspace(3.0, 9.0, reverse_idx.sum())
    rev["hr"] = np.linspace(82.0, 88.0, reverse_idx.sum())
    dx = np.diff(rev["lon"], prepend=rev["lon"].iloc[0])
    dy = np.diff(rev["lat"], prepend=rev["lat"].iloc[0])
    rev["heading"] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    out.loc[reverse_idx, ["lat", "lon", "heading", "speed", "hr"]] = rev[["lat", "lon", "heading", "speed", "hr"]].to_numpy()
    return out


def get_source_bundle(location_key: str, trace_type: str):
    is_real = location_key == "us101" or trace_type.startswith("real_")
    injected = "injected" if "real_" in trace_type and trace_type != "real_normal" else "native"
    source_keys = {"chennai": "anna_salai", "la": "i405", "dc": "i95", "us101": "us101"}
    tower_keys = {"chennai": "anna_salai", "la": "i405", "dc": "i95", "us101": "us101_ngsim"}
    geo_key = source_keys.get(location_key, location_key)
    tower_key = tower_keys.get(location_key, location_key)
    geometry_source = "OSM/Overpass" if load_overpass_features(geo_key) else "synthetic geojson"
    tower_source = "OpenCelliD" if load_real_towers(tower_key) else "synthetic tower map"
    workzone_source = "WZDx feed" if load_workzone_events(geo_key) else "synthetic/no feed"
    curated_manifest = _curated_manifest_path("us101")
    curated_count = 0
    if curated_manifest.exists():
        curated_count = json.loads(curated_manifest.read_text()).get("count", 0)
    return {
        "trace_source": "FHWA NGSIM" if is_real else "Synthetic generator",
        "raw_gps_source": "Microsoft GeoLife" if has_raw_gps_data() else "not staged",
        "real_labeled_events": "staged" if has_real_labeled_events() else "not staged",
        "trace_variant": injected if is_real else "synthetic",
        "geometry_source": geometry_source,
        "tower_source": tower_source,
        "workzone_source": workzone_source,
        "curated_real_traces": curated_count,
    }
