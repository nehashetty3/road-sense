from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import REAL_CASES, ROBUSTNESS_CASES, evaluate_trace, run_robustness_suite, score_predictions
from real_data import (
    get_source_bundle,
    has_raw_gps_data,
    has_real_labeled_events,
    has_real_ngsim_data,
    has_real_workzone_data,
)


RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _tag_record(record):
    trace_type = record["trace_type"]
    tags = []
    if trace_type.startswith("real_"):
        tags.append("real")
    else:
        tags.append("synthetic")
    if "construction" in trace_type:
        tags.append("construction")
    if "wrongway" in trace_type:
        tags.append("wrongway")
    if any(token in trace_type for token in ["lane_confused", "stop_reverse", "fault", "drunk"]):
        tags.append("edge_case")
    return tags


def _subset_metrics(records, pred_key):
    return score_predictions(records, pred_key) if records else {}


def build_report():
    result_files = {
        "baseline_pred": RESULTS_DIR / "baseline_pred.json",
        "lnn_ode_pred": RESULTS_DIR / "lnn_ode_pred.json",
        "full_stack_pred": RESULTS_DIR / "full_stack_pred.json",
    }
    payloads = {name: json.loads(path.read_text()) for name, path in result_files.items() if path.exists()}
    if len(payloads) != len(result_files):
        raise FileNotFoundError("Run the benchmark experiments before generating the breakdown report.")

    core_records = payloads["full_stack_pred"]["records"]
    robustness_records = run_robustness_suite()
    source_bundle = get_source_bundle("us101", "real_normal")

    by_model = {}
    for name, payload in payloads.items():
        pred_key = payload["pred_key"]
        records = payload["records"]
        real_records = [r for r in records if "real" in _tag_record(r)]
        synthetic_records = [r for r in records if "synthetic" in _tag_record(r)]
        construction_records = [r for r in records if "construction" in _tag_record(r)]
        wrongway_records = [r for r in records if "wrongway" in _tag_record(r)]
        by_model[name] = {
            "summary": payload["summary"],
            "synthetic_only": _subset_metrics(synthetic_records, pred_key),
            "real_only": _subset_metrics(real_records, pred_key),
            "construction_only": _subset_metrics(construction_records, pred_key),
            "wrongway_only": _subset_metrics(wrongway_records, pred_key),
            "per_case": records,
        }

    robustness_by_model = {}
    for name, payload in payloads.items():
        pred_key = payload["pred_key"]
        robustness_by_model[name] = {
            "summary": _subset_metrics(robustness_records, pred_key),
            "records": robustness_records,
        }

    report = {
        "core_real_cases": [{"loc": loc, "trace_type": trace_type, "label": label} for loc, trace_type, _, _, label in REAL_CASES],
        "robustness_cases": [{"loc": loc, "trace_type": trace_type, "label": label} for loc, trace_type, _, _, label in ROBUSTNESS_CASES],
        "source_bundle": source_bundle,
        "has_real_ngsim": has_real_ngsim_data(),
        "has_raw_gps": has_raw_gps_data(),
        "has_real_labeled_events": has_real_labeled_events(),
        "has_real_workzone_feed": has_real_workzone_data(),
        "validation_tier": (
            "partner_validated"
            if has_real_ngsim_data() and has_raw_gps_data() and has_real_workzone_data() and has_real_labeled_events()
            else "prototype_plus_external_feeds"
            if has_real_ngsim_data() and has_raw_gps_data() and has_real_workzone_data()
            else "prototype_demo_validation"
        ),
        "models": by_model,
        "robustness": robustness_by_model,
        "notes": [
            "Core ablation metrics come from the saved benchmark JSON files.",
            "Robustness cases extend the real US-101 trace with short wrong-way, lane-confusion, and stop-reverse edge cases.",
            "Scores are prototype validation until raw GPS, real labeled events, OpenCelliD, and live WZDx assets are staged.",
            "OpenCelliD remains credential-gated until user credentials are supplied.",
        ],
    }
    return report


def write_outputs(report):
    json_path = RESULTS_DIR / "report_breakdown.json"
    md_path = RESULTS_DIR / "report_breakdown.md"
    json_path.write_text(json.dumps(report, indent=2))

    full = report["models"]["full_stack_pred"]
    real = full["real_only"]
    robust = report["robustness"]["full_stack_pred"]["summary"]
    lines = [
        "# SwarmMind Evaluation Breakdown",
        "",
        "## Full TTI-Stack",
        f"- Validation tier: {report['validation_tier']}",
        f"- Overall: F1={full['summary']['f1']:.4f}, FPR={full['summary']['false_positive_rate']:.4f}",
        f"- Real-only core: F1={real.get('f1', 0.0):.4f}, FPR={real.get('false_positive_rate', 0.0):.4f}",
        f"- Robustness suite: F1={robust.get('f1', 0.0):.4f}, FPR={robust.get('false_positive_rate', 0.0):.4f}",
        "",
        "## Source Provenance",
        f"- Trace source: {report['source_bundle']['trace_source']}",
        f"- Raw GPS source: {report['source_bundle']['raw_gps_source']}",
        f"- Real labeled events: {report['source_bundle']['real_labeled_events']}",
        f"- Geometry source: {report['source_bundle']['geometry_source']}",
        f"- Connectivity source: {report['source_bundle']['tower_source']}",
        f"- Work-zone source: {report['source_bundle']['workzone_source']}",
        f"- Curated real traces: {report['source_bundle']['curated_real_traces']}",
        "",
        "## Notes",
    ]
    lines.extend(f"- {note}" for note in report["notes"])
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


if __name__ == "__main__":
    report = build_report()
    json_path, md_path = write_outputs(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
