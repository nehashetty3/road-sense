from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import REAL_CASES, ROBUSTNESS_CASES, evaluate_trace
from layers.tti_filter import fit_tti_filter, save_tti_weights


RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_samples():
    cases = []
    for loc in ["chennai", "la", "dc"]:
        for trace_type in ["normal", "drunk", "fault", "wrongway", "construction"]:
            label = 1 if trace_type == "wrongway" else 0
            cases.append((loc, trace_type, label))
    for _, trace_type, _, _, label in REAL_CASES + ROBUSTNESS_CASES:
        cases.append(("us101", trace_type, label))

    samples = []
    for loc, trace_type, label in cases:
        record = evaluate_trace(loc, trace_type)
        samples.append({
            "label": label,
            "ode_term": record["tti_ode_term"],
            "topology_term": record["tti_topology_term"],
            "surprise_term": record["tti_surprise_term"],
            "noise_penalty": record["tti_noise_penalty"],
            "loc": loc,
            "trace_type": trace_type,
        })
    return samples


if __name__ == "__main__":
    samples = build_samples()
    weights = fit_tti_filter(samples)
    save_tti_weights(weights)
    out_path = RESULTS_DIR / "tti_filter_calibration.json"
    out_path.write_text(json.dumps({"weights": weights, "samples": samples}, indent=2))
    print(f"Wrote {out_path}")
