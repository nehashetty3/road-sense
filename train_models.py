"""
SwarmMind Advanced Model Trainer
Trains and caches all ML models: LNN, KAN, ODE, RSSM.
Safe to call multiple times (skips if weights already exist).
"""

import pandas as pd
import json
from pathlib import Path
import time
from real_data import has_real_ngsim_data, load_curated_real_traces, load_real_ngsim_trace

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

LOCS = {
    "chennai": {"center": (13.049, 80.2518), "direction": 180.0},
    "la":      {"center": (33.9206, -118.3462), "direction": 350.0},
    "dc":      {"center": (38.8051, -77.0468), "direction": 45.0},
}
TYPES = ["normal", "drunk", "fault", "wrongway"]


def _load(loc, ttype):
    return pd.read_csv(DATA_DIR / f"{ttype}_{loc}.csv")


def _load_geo(loc):
    with open(DATA_DIR / f"{loc}.geojson") as f:
        return json.load(f)["features"]


def train_all(progress_cb=None, force_retrain=False):
    """
    Train all advanced models. progress_cb(message, pct) for UI updates.
    Returns dict of status messages.
    """
    def prog(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)
        else:
            print(f"[{pct:3d}%] {msg}")

    results = {}

    # ── LNN ──────────────────────────────────────────────────────────────────
    from layers.f1_lnn import load_or_train_lnn
    prog("Training Liquid Neural Network (LNN)…", 5)
    t0 = time.time()
    traces_dict = {}
    rd_dict = {}
    for loc, meta in LOCS.items():
        traces_dict[loc] = {tt: _load(loc, tt) for tt in TYPES}
        rd_dict[loc] = meta["direction"]
    if has_real_ngsim_data():
        real_traces = load_curated_real_traces("us101", max_traces=4)
        for idx, trace in enumerate(real_traces):
            key = f"us101_trace_{idx + 1}"
            traces_dict[key] = {
                "normal": trace.copy(),
                "wrongway": load_real_ngsim_trace("us101", "wrongway_injected", trace_index=idx),
                "construction": load_real_ngsim_trace("us101", "construction_injected", trace_index=idx),
                "short_wrongway": load_real_ngsim_trace("us101", "short_wrongway_injected", trace_index=idx),
                "lane_confused": load_real_ngsim_trace("us101", "lane_confused_injected", trace_index=idx),
                "stop_reverse": load_real_ngsim_trace("us101", "stop_reverse_injected", trace_index=idx),
            }
            rd_dict[key] = 130.0
    lnn_model, lnn_losses = load_or_train_lnn(traces_dict, rd_dict,
                                               force_retrain=force_retrain)
    results["lnn"] = {"ok": True, "time": round(time.time() - t0, 1),
                      "epochs": len(lnn_losses), "final_loss": lnn_losses[-1] if lnn_losses else 0}
    prog(f"LNN trained  ({results['lnn']['time']}s)", 18)



    # ── Neural ODE ───────────────────────────────────────────────────────────
    from layers.f5_f6_f9 import load_or_train_ode
    prog("Training Neural ODE dynamics model…", 30)
    t0 = time.time()
    normal_df = _load("chennai", "normal")
    ode_model, ode_mins, ode_ranges = load_or_train_ode(normal_df,
                                                         force_retrain=force_retrain)
    results["ode"] = {"ok": True, "time": round(time.time() - t0, 1)}
    prog(f"Neural ODE trained  ({results['ode']['time']}s)", 50)

    # ── RSSM ─────────────────────────────────────────────────────────────────
    from layers.f8_f10 import load_or_train_rssm
    prog("Training World Model / RSSM…", 55)
    t0 = time.time()
    rssm_model, rssm_mins, rssm_ranges = load_or_train_rssm(
        normal_df, force_retrain=force_retrain)
    results["rssm"] = {"ok": True, "time": round(time.time() - t0, 1)}
    prog(f"RSSM trained  ({results['rssm']['time']}s)", 70)

    # ── KAN ──────────────────────────────────────────────────────────────────
    from layers.f4_kan import (load_or_train_kan, build_training_dataset,
                                KANClassifier)
    from layers.f2_topology import compute_topology_fingerprint
    from layers.core import run_all_layers
    geo = _load_geo("chennai")
    with open(DATA_DIR / "towers_chennai.json") as f:
        towers = json.load(f)
    with open(DATA_DIR / "swarm_chennai.json") as f:
        swarm = json.load(f)

    prog("Building KAN training dataset…", 75)
    t0 = time.time()
    all_l1, all_l3, all_topo, labels = [], [], [], []
    for tt in TYPES:
        df = _load("chennai", tt)
        r = run_all_layers(df, geo, towers, swarm,
                           LOCS["chennai"]["direction"], 50)
        all_l1.append(r["layer1"])
        all_l3.append(r["layer3"])
        topo = compute_topology_fingerprint(df)
        all_topo.append(topo)
        labels.append(tt)

    prog("Training KAN decision layer…", 88)
    from layers.f4_kan import build_training_dataset, train_kan
    X, y = build_training_dataset(all_l1, all_l3, all_topo, labels)
    if not force_retrain and (DATA_DIR / "kan_weights.pt").exists():
        kan_model = load_or_train_kan()
    else:
        kan_model = load_or_train_kan(X, y, force_retrain=True)

    results["kan"] = {"ok": True, "time": round(time.time() - t0, 1)}
    prog(f"KAN trained  ({results['kan']['time']}s)", 96)

    prog("All models ready!", 100)
    return results


if __name__ == "__main__":
    r = train_all(force_retrain=False)
    print("\n=== Training Summary ===")
    for model, info in r.items():
        print(f"  {model:15}: {'OK' if info['ok'] else 'FAILED'}  {info.get('time', '?')}s")
