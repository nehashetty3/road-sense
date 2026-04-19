#!/usr/bin/env python3
"""
SwarmMind Data Generator
Produces 9 GPS trace CSVs (normal/drunk/fault/wrongway x chennai/la/dc),
3 road GeoJSONs, and 3 cell-tower JSON files for the demo.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

np.random.seed(42)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LOCATIONS = {
    "chennai": {
        "name": "Anna Salai, Chennai",
        "center": (13.0490, 80.2518),
        "base_heading": 180.0,
        "speed_kmh": 45.0,
        "highway": "primary",
        "road_name": "Anna Salai",
    },
    "la": {
        "name": "I-405, Los Angeles",
        "center": (33.9206, -118.3462),
        "base_heading": 350.0,
        "speed_kmh": 65.0,
        "highway": "motorway",
        "road_name": "I-405",
    },
    "dc": {
        "name": "I-95, Washington DC",
        "center": (38.8051, -77.0468),
        "base_heading": 45.0,
        "speed_kmh": 70.0,
        "highway": "motorway",
        "road_name": "I-95",
    },
}

TYPES = ["normal", "drunk", "fault", "wrongway", "construction"]
N = 300  # 300 seconds at 1 Hz


def gps_step(lat, lon, heading_deg, speed_kmh):
    heading_rad = np.radians(heading_deg)
    dist_km = speed_kmh / 3600.0
    d_lat = dist_km * np.cos(heading_rad) / 111.0
    d_lon = dist_km * np.sin(heading_rad) / (111.0 * np.cos(np.radians(lat)))
    return lat + d_lat, lon + d_lon


def generate_trace(loc_key, trace_type):
    loc = LOCATIONS[loc_key]
    clat, clon = loc["center"]
    bh = loc["base_heading"]
    bs = loc["speed_kmh"]

    # Start the vehicle a bit before centre
    br = np.radians(bh + 180)
    lat = clat + 0.008 * np.cos(br)
    lon = clon + 0.008 * np.sin(br)

    rows = []
    for t in range(N):
        if trace_type == "normal":
            h = bh + np.random.normal(0, 2.5)
            s = bs + np.random.normal(0, 2.5)
        elif trace_type == "drunk":
            h = bh + 18 * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 2)
            s = bs + 6 * np.abs(np.sin(2 * np.pi * 0.2 * t)) + np.random.normal(0, 2)
        elif trace_type == "fault":
            h = bh + 12 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 1)
            s = max(10.0, bs - 0.08 * t + np.random.normal(0, 1.5))
        elif trace_type == "construction":
            if 110 <= t <= 165:
                diversion = 130 if t < 138 else -120
                h = bh + diversion + np.random.normal(0, 5)
                s = max(15.0, bs * 0.55 + np.random.normal(0, 2.5))
            else:
                h = bh + np.random.normal(0, 3.0)
                s = bs * 0.92 + np.random.normal(0, 2.0)
        else:  # wrongway
            h = (bh + 180) % 360 + np.random.normal(0, 3)
            s = bs * 0.75 + np.random.normal(0, 3)

        h = h % 360
        s = max(5.0, s)

        # Heart-rate proxy for rPPG layer
        if trace_type in ("wrongway", "drunk"):
            hr = 78 + 40 * (t / N) + 8 * np.sin(2 * np.pi * 0.06 * t) + np.random.normal(0, 4)
        elif trace_type == "construction":
            hr = 74 + 6 * np.sin(2 * np.pi * 0.04 * t) + np.random.normal(0, 2.5)
        else:
            hr = 70 + 5 * np.sin(2 * np.pi * 0.03 * t) + np.random.normal(0, 2.5)
        hr = float(np.clip(hr, 50, 165))

        rows.append({"time": float(t), "lat": lat, "lon": lon,
                     "speed": round(s, 2), "heading": round(h, 2), "hr": round(hr, 1)})
        lat, lon = gps_step(lat, lon, h, s)

    return pd.DataFrame(rows)


def generate_geojson(loc_key):
    loc = LOCATIONS[loc_key]
    clat, clon = loc["center"]
    bh = loc["base_heading"]
    hr = np.radians(bh)
    feats = []

    # Main road segments (10 segments along the road)
    seg = 0.004  # ~400 m per segment
    for i in range(-5, 6):
        p1_lat = clat + i * seg * np.cos(hr)
        p1_lon = clon + i * seg * np.sin(hr)
        p2_lat = clat + (i + 1) * seg * np.cos(hr)
        p2_lon = clon + (i + 1) * seg * np.sin(hr)
        feats.append({
            "type": "Feature",
            "properties": {
                "highway": loc["highway"],
                "name": loc["road_name"],
                "oneway": True,
                "direction": bh,
                "construction": False,
                "cycleway": False,
                "lanes": 3 if loc["highway"] == "motorway" else 2,
            },
            "geometry": {"type": "LineString",
                         "coordinates": [[p1_lon, p1_lat], [p2_lon, p2_lat]]},
        })

    # Construction zone segment
    pr = hr + np.pi / 8  # slightly off road
    feats.append({
        "type": "Feature",
        "properties": {
            "highway": loc["highway"], "name": loc["road_name"],
            "oneway": True, "direction": bh,
            "construction": True, "cycleway": False, "lanes": 1,
        },
        "geometry": {"type": "LineString",
                     "coordinates": [
                         [clon + 0.006 * np.sin(hr), clat + 0.006 * np.cos(hr)],
                         [clon + 0.010 * np.sin(hr), clat + 0.010 * np.cos(hr)],
                     ]},
    })

    # Parallel cycleway
    perp = hr + np.pi / 2
    cy_lat = 0.0012 * np.cos(perp)
    cy_lon = 0.0012 * np.sin(perp)
    feats.append({
        "type": "Feature",
        "properties": {
            "highway": "cycleway", "name": f"{loc['road_name']} cycleway",
            "oneway": False, "direction": bh,
            "construction": False, "cycleway": True, "lanes": 1,
        },
        "geometry": {"type": "LineString",
                     "coordinates": [
                         [clon + cy_lon - 0.012 * np.sin(hr),
                          clat + cy_lat - 0.012 * np.cos(hr)],
                         [clon + cy_lon + 0.012 * np.sin(hr),
                          clat + cy_lat + 0.012 * np.cos(hr)],
                     ]},
    })

    return {"type": "FeatureCollection", "features": feats}


def generate_towers(loc_key):
    loc = LOCATIONS[loc_key]
    clat, clon = loc["center"]
    towers = []
    n_types = [("5G", 0.35), ("4G", 0.50), ("3G", 0.15)]
    for _ in range(25):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.002, 0.025)
        t_type = np.random.choice([x[0] for x in n_types],
                                   p=[x[1] for x in n_types])
        sig = {"5G": np.random.uniform(0.75, 1.0),
               "4G": np.random.uniform(0.45, 0.85),
               "3G": np.random.uniform(0.2, 0.55)}[t_type]
        towers.append({
            "lat": clat + r * np.sin(angle),
            "lon": clon + r * np.cos(angle),
            "type": t_type,
            "signal": round(sig, 3),
        })
    # Inject a dead zone
    towers.append({"lat": clat + 0.008, "lon": clon + 0.008,
                   "type": "DEAD", "signal": 0.0})
    return towers


def generate_swarm_vehicles(loc_key, n=500):
    """Background swarm vehicles (normal traffic) for V2X layer."""
    loc = LOCATIONS[loc_key]
    clat, clon = loc["center"]
    bh = loc["base_heading"]
    vehicles = []
    for i in range(n):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.001, 0.025)
        vehicles.append({
            "id": i,
            "lat": clat + r * np.sin(angle),
            "lon": clon + r * np.cos(angle),
            "heading": bh + np.random.normal(0, 5),
            "speed": loc["speed_kmh"] + np.random.normal(0, 8),
        })
    return vehicles


if __name__ == "__main__":
    print("SwarmMind — generating synthetic dataset…\n")
    for loc_key in LOCATIONS:
        for ttype in TYPES:
            df = generate_trace(loc_key, ttype)
            path = DATA_DIR / f"{ttype}_{loc_key}.csv"
            df.to_csv(path, index=False)
            print(f"  ✓  {path}  ({len(df)} rows)")

        geo = generate_geojson(loc_key)
        gpath = DATA_DIR / f"{loc_key}.geojson"
        with open(gpath, "w") as f:
            json.dump(geo, f)
        print(f"  ✓  {gpath}  ({len(geo['features'])} features)")

        towers = generate_towers(loc_key)
        tpath = DATA_DIR / f"towers_{loc_key}.json"
        with open(tpath, "w") as f:
            json.dump(towers, f)
        print(f"  ✓  {tpath}  ({len(towers)} towers)")

        swarm = generate_swarm_vehicles(loc_key)
        spath = DATA_DIR / f"swarm_{loc_key}.json"
        with open(spath, "w") as f:
            json.dump(swarm, f)
        print(f"  ✓  {spath}  ({len(swarm)} vehicles)\n")

    print("Done. All data written to /data/")
