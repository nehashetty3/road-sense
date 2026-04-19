from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from layers.core import run_all_layers, _heading_delta
from layers.baseline import fft_rppg_baseline
from layers.f1_lnn import load_or_train_lnn, lnn_predict_trace
from layers.f2_topology import compute_topology_fingerprint
from layers.f5_f6_f9 import load_or_train_ode, ode_reconstruction_error, SurpriseScoreModel
from layers.tti_filter import compute_tti_filter
from real_data import (
    has_real_ngsim_data,
    load_curated_real_traces,
    load_real_ngsim_trace,
    load_overpass_features,
    load_real_towers,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOCS = {"chennai": 180.0, "la": 350.0, "dc": 45.0}
TRACE_TYPES = ["normal", "drunk", "fault", "wrongway", "construction"]
REAL_CASES = [
    ("us101_real", "real_normal", 130.0, "us101", 0),
    ("us101_real", "real_wrongway", 130.0, "us101", 1),
    ("us101_real", "real_construction", 130.0, "us101", 0),
]
ROBUSTNESS_CASES = [
    ("us101_robustness", "real_short_wrongway", 130.0, "us101", 1),
    ("us101_robustness", "real_lane_confused", 130.0, "us101", 0),
    ("us101_robustness", "real_stop_reverse", 130.0, "us101", 0),
]
_LNN_CACHE = {}
_ODE_CACHE = None
_TRACE_CACHE = {}


def ensure_construction_data():
    from generate_data import generate_trace

    for loc in LOCS:
        path = DATA_DIR / f"construction_{loc}.csv"
        if not path.exists():
            return False
    return True


def load_trace(loc: str, trace_type: str) -> pd.DataFrame:
    cache_key = (loc, trace_type)
    if cache_key in _TRACE_CACHE:
        return _TRACE_CACHE[cache_key].copy()
    if loc == "us101" and trace_type == "normal":
        df = load_real_ngsim_trace("us101", "normal")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "real_normal":
        df = load_real_ngsim_trace("us101", "normal")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "real_wrongway":
        df = load_real_ngsim_trace("us101", "wrongway_injected")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "real_construction":
        df = load_real_ngsim_trace("us101", "construction_injected")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "real_short_wrongway":
        df = load_real_ngsim_trace("us101", "short_wrongway_injected")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "real_lane_confused":
        df = load_real_ngsim_trace("us101", "lane_confused_injected")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "real_stop_reverse":
        df = load_real_ngsim_trace("us101", "stop_reverse_injected")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    path = DATA_DIR / f"{trace_type}_{loc}.csv"
    if path.exists():
        df = pd.read_csv(path)
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    if trace_type == "construction":
        from generate_data import generate_trace
        df = generate_trace(loc, "construction")
        _TRACE_CACHE[cache_key] = df
        return df.copy()
    df = pd.read_csv(path)
    _TRACE_CACHE[cache_key] = df
    return df.copy()


def load_geo(loc: str):
    if loc == "us101":
        return load_overpass_features("us101")
    with open(DATA_DIR / f"{loc}.geojson") as handle:
        return json.load(handle)["features"]


def load_towers(loc: str):
    if loc == "us101":
        towers = load_real_towers("us101_ngsim")
        if towers:
            return towers
        with open(DATA_DIR / "towers_la.json") as handle:
            return json.load(handle)
    with open(DATA_DIR / f"towers_{loc}.json") as handle:
        return json.load(handle)


def load_swarm(loc: str):
    if loc == "us101":
        with open(DATA_DIR / "swarm_la.json") as handle:
            return json.load(handle)
    with open(DATA_DIR / f"swarm_{loc}.json") as handle:
        return json.load(handle)


def build_lnn_traces():
    traces = {
        loc: {tt: load_trace(loc, tt) for tt in ["normal", "drunk", "fault", "wrongway"]}
        for loc in LOCS
    }
    road_dirs = dict(LOCS)
    if has_real_ngsim_data():
        real_traces = load_curated_real_traces("us101", max_traces=4)
        if real_traces:
            for idx, df in enumerate(real_traces):
                key = f"us101_trace_{idx + 1}"
                _TRACE_CACHE[(key, "normal")] = df
                traces[key] = {
                    "normal": df.copy(),
                    "wrongway": load_real_ngsim_trace("us101", "wrongway_injected", trace_index=idx),
                    "construction": load_real_ngsim_trace("us101", "construction_injected", trace_index=idx),
                    "short_wrongway": load_real_ngsim_trace("us101", "short_wrongway_injected", trace_index=idx),
                    "lane_confused": load_real_ngsim_trace("us101", "lane_confused_injected", trace_index=idx),
                    "stop_reverse": load_real_ngsim_trace("us101", "stop_reverse_injected", trace_index=idx),
                }
                road_dirs[key] = 130.0
    return traces, road_dirs


def load_lnn(force_retrain=False, seed=None, save_weights=True):
    cache_key = (force_retrain, seed, save_weights)
    if not force_retrain and cache_key in _LNN_CACHE:
        return _LNN_CACHE[cache_key]
    traces, road_dirs = build_lnn_traces()
    model, _ = load_or_train_lnn(traces, road_dirs, force_retrain=force_retrain, seed=seed, save_weights=save_weights)
    if not force_retrain:
        _LNN_CACHE[cache_key] = model
    return model


def load_ode():
    global _ODE_CACHE
    if _ODE_CACHE is not None:
        return _ODE_CACHE
    normal_df = load_trace("chennai", "normal")
    _ODE_CACHE = load_or_train_ode(normal_df)
    return _ODE_CACHE


def construction_noise(df: pd.DataFrame, road_direction: float) -> float:
    deltas = np.array([_heading_delta(h, road_direction) for h in df["heading"].values], dtype=float)
    return float(np.std(np.diff(deltas, prepend=deltas[0])) / 180.0)


def evaluate_trace(loc: str, trace_type: str):
    road_direction = 130.0 if loc == "us101" else LOCS.get(loc, 180.0)
    df = load_trace(loc, trace_type)
    geo = load_geo(loc)
    towers = load_towers(loc)
    swarm = load_swarm(loc)
    pipeline = run_all_layers(df, geo, towers, swarm, road_direction, 75)

    lnn_model = load_lnn()
    lnn_res = lnn_predict_trace(lnn_model, df, threshold=0.30, road_direction=road_direction)
    ode_model, mins, ranges = load_ode()
    ode_res = ode_reconstruction_error(ode_model, mins, ranges, df)
    topo_res = compute_topology_fingerprint(df)
    surprise = SurpriseScoreModel()
    surprise.fit(load_trace(loc, "normal"))
    surprise_res = surprise.score_trace(df)
    tti_res = compute_tti_filter(
        ode_res["max_mse"],
        topo_res["max_persistence"],
        surprise_res["max_accumulated"],
        construction_noise(df, road_direction),
    )

    baseline_res = fft_rppg_baseline(df["heading"].values, df["hr"].values, road_direction)
    baseline_flag = baseline_res["predicted_wrong_way"]
    real_case = trace_type.startswith("real_")
    lnn_threshold = 0.27 if real_case else 0.28
    min_wrong_count = 20 if real_case else 40
    full_stack_wrong_count = 20 if real_case else 120
    lnn_ode_flag = baseline_flag and lnn_res["max_score"] >= lnn_threshold and pipeline["layer1"]["wrong_count"] >= min_wrong_count
    full_stack_flag = (
        lnn_ode_flag
        and tti_res["confidence"] >= 0.55
        and pipeline["layer1"]["wrong_count"] >= full_stack_wrong_count
        and len(pipeline["final_events"]) >= 8
    )

    return {
        "loc": loc,
        "trace_type": trace_type,
        "label": 1 if trace_type == "wrongway" else 0,
        "baseline_pred": int(baseline_flag),
        "lnn_ode_pred": int(lnn_ode_flag),
        "full_stack_pred": int(full_stack_flag),
        "baseline_hits": baseline_res["sustained_hits"],
        "lnn_max": lnn_res["max_score"],
        "ode_max_mse": ode_res["max_mse"],
        "topology": topo_res["max_persistence"],
        "surprise_bits": surprise_res["max_accumulated"],
        "tti_confidence": tti_res["confidence"],
        "tti_ode_term": tti_res["ode_term"],
        "tti_topology_term": tti_res["topology_term"],
        "tti_surprise_term": tti_res["surprise_term"],
        "tti_noise_penalty": tti_res["noise_penalty"],
        "tti_topology_weight": tti_res["topology_weight"],
        "wrong_count": pipeline["layer1"]["wrong_count"],
        "construction_downgrades": pipeline["layer7"]["downgraded_count"],
    }


def score_predictions(records, pred_key: str):
    y_true = np.array([r["label"] for r in records], dtype=int)
    y_pred = np.array([r[pred_key] for r in records], dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fpr = fp / max(1, fp + tn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
    }


def run_suite(pred_key: str, label: str):
    records = [evaluate_trace(loc, tt) for loc in LOCS for tt in TRACE_TYPES]
    if has_real_ngsim_data():
        for loc, trace_type, road_direction, _, expected in REAL_CASES:
            rec = evaluate_trace("us101", trace_type)
            rec["label"] = expected
            rec["loc"] = loc
            records.append(rec)
    summary = score_predictions(records, pred_key)
    payload = {"label": label, "pred_key": pred_key, "summary": summary, "records": records}
    out_path = RESULTS_DIR / f"{pred_key}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def run_robustness_suite():
    records = []
    if has_real_ngsim_data():
        for loc, trace_type, _, _, expected in ROBUSTNESS_CASES:
            rec = evaluate_trace("us101", trace_type)
            rec["label"] = expected
            rec["loc"] = loc
            records.append(rec)
    return records
