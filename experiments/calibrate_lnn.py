#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import load_lnn, run_suite


OUT_PATH = Path("/tmp/swarmmind_lnn_calibration.json")
SEEDS = [7, 21]


def main():
    rounds = []
    best = None
    baseline = run_suite("baseline_pred", "Baseline (FFT + rPPG)")["summary"]
    for seed in SEEDS:
        print(f"[seed={seed}] retraining LNN with real US-101 traces included", flush=True)
        load_lnn(force_retrain=True, seed=seed, save_weights=True)
        lnn_ode = run_suite("lnn_ode_pred", "+ LNN + ODE")["summary"]
        full = run_suite("full_stack_pred", "Full TTI-Stack")["summary"]
        row = {
            "seed": seed,
            "baseline": baseline,
            "lnn_ode": lnn_ode,
            "full_stack": full,
        }
        rounds.append(row)
        score = (full["f1"], full["recall"], -full["false_positive_rate"])
        if best is None or score > best["score"]:
            best = {"score": score, "seed": seed, "full_stack": full, "lnn_ode": lnn_ode}
        print(f"[seed={seed}] full_stack={full} lnn_ode={lnn_ode}", flush=True)

    summary = {
        "rounds": rounds,
        "best_seed": best["seed"] if best else None,
        "full_stack_f1_mean": round(statistics.mean(r["full_stack"]["f1"] for r in rounds), 4),
        "full_stack_f1_std": round(statistics.pstdev(r["full_stack"]["f1"] for r in rounds), 4),
        "lnn_ode_f1_mean": round(statistics.mean(r["lnn_ode"]["f1"] for r in rounds), 4),
        "lnn_ode_f1_std": round(statistics.pstdev(r["lnn_ode"]["f1"] for r in rounds), 4),
    }
    payload = {"summary": summary, "best": best, "rounds": rounds}
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved calibration summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
