"""
Simple rule-based baselines for wrong-way detection.
These are intentionally minimal and exist to anchor the ablation story.
"""

from __future__ import annotations

from typing import Iterable


def heading_delta_deg(heading: float, road_direction: float) -> float:
    diff = abs(float(heading) - float(road_direction)) % 360
    return min(diff, 360 - diff)


def heading_threshold_detector(
    headings: Iterable[float],
    road_direction: float,
    threshold_deg: float = 100.0,
    min_hits: int = 3,
) -> dict:
    deltas = [heading_delta_deg(h, road_direction) for h in headings]
    hits = [d > threshold_deg for d in deltas]
    sustained_hits = sum(hits)
    return {
        "threshold_deg": threshold_deg,
        "sustained_hits": sustained_hits,
        "max_delta": round(max(deltas) if deltas else 0.0, 3),
        "mean_delta": round(sum(deltas) / max(1, len(deltas)), 3),
        "predicted_wrong_way": sustained_hits >= min_hits,
    }


def fft_rppg_baseline(
    headings: Iterable[float],
    heart_rates: Iterable[float],
    road_direction: float,
    threshold_deg: float = 100.0,
    stress_bpm: float = 110.0,
    min_hits: int = 3,
) -> dict:
    heading_result = heading_threshold_detector(headings, road_direction, threshold_deg, min_hits)
    hr_values = [float(v) for v in heart_rates]
    max_hr = max(hr_values) if hr_values else 0.0
    return {
        **heading_result,
        "max_hr": round(max_hr, 2),
        "stress_flag": max_hr > stress_bpm,
        "predicted_wrong_way": heading_result["predicted_wrong_way"] or max_hr > stress_bpm,
    }
