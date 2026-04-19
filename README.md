# SwarmMind
### 9-Layer Safety Engine + TTI Safety Stack for Wrong-Way Detection

SwarmMind is a prototype vehicular-anomaly-detection system that uses GPS traces, open-map priors, and lightweight V2X logic to warn nearby vehicles before a wrong-way encounter becomes unavoidable. The core idea is simple: detect counter-flow early, confirm it with multiple independent anomaly signals, suppress construction-zone false positives, then fan the alert across the road network.

## Abstract
SwarmMind is a TTI-Safety-Stack prototype for vehicular anomaly detection. It combines a 9-layer safety engine with adaptive sequence modeling, continuous-time dynamics, topology-aware anomaly scoring, interpretable KAN decisions, spectral road vulnerability, and V2X alert routing. The stack is intentionally modular: each block emits one scalar or vector, and the system still functions if any single advanced block is unavailable. Future work can plug LiDAR features, radar tracks, or V2X CAM streams into the same pipeline.

Current repo note:
The live demo is reproducible from local synthetic traces, real FHWA NGSIM `US-101` background traces, real OSM/Overpass corridor geometry, and injected real-background stress cases. The benchmark chart reads saved experiment JSON so the app, README, and judge story use the same artifacts.

## What It Does
- Layer 1 checks whether GPS heading agrees with road direction and peer flow.
- Layer 2 routes alerts to private and public channels.
- Layer 3 separates wrong-way motion from drunk swerving and mechanical drift.
- Layer 4 uses rPPG-derived stress as a demo-only confidence boost.
- Layer 5 predicts collision zones and time-to-impact hotspots.
- Layers 6 and 7 suppress false positives from cycleways and construction diversions.
- Layer 8 ripples alerts across nearby vehicles.
- Layer 9 checks connectivity and recommends Bluetooth fallback if coverage degrades.

## TTI Stack
The advanced research-facing stack used in the live story is:

`GPS -> LNN -> Neural ODE -> TDA -> Surprise Bits -> TTI-Filter -> KAN -> Spectral Vulnerability -> V2X`

### TTI-Filter
`TTI-Filter` is the named lightweight module added for this iteration. It takes:
- ODE reconstruction error
- TDA persistence
- Surprise bits
- Trajectory noise penalty

It outputs a confidence re-weighting that decides how much the topology score should be trusted under noisy conditions. This is the mechanism that makes the construction-zone stress test feel like a true systems contribution instead of just a pile of metrics.

## Related Work
- LNN traffic modeling: adaptive continuous-time recurrent models can absorb changing traffic regimes without per-segment retraining.
- TDA trajectory anomaly work: persistence summaries give structural signals for rare path behaviors.
- Neural ODE traffic dynamics: continuous latent flows are useful for manifold reconstruction and anomaly scoring.
- KAN interpretability work: spline-like feature curves make neural decisions easier to audit.

Positioning:
SwarmMind composes these ideas into a new TTI-Safety-Stack for wrong-way detection and false-positive suppression.

## New Evaluation Angle
This version introduces a synthetic but explicit `wrong-way vs construction-zone diversion` stress case, a failure mode that is often under-discussed in traffic safety demos. The point is not only to detect wrong-way motion, but to show that the stack suppresses legal construction detours that would fool a simpler heading-only trigger.

## 10/10 Improvement Pass
The latest pass adds the ten remaining credibility upgrades:
- Real construction-zone truth path: WZDx GeoJSON feeds can be fetched into `external_data/wzdx/` and overlaid in the app.
- Real connectivity path: OpenCelliD imports remain supported without faking credentials; the app labels synthetic vs real tower context.
- Real anomaly diversity: real `US-101` traces now support sustained wrong-way, short wrong-way, construction diversion, lane-confused merge, and stop-reverse edge cases.
- Calibrated confidence fusion: `TTI-Filter` has a trainable logistic calibration layer saved in `data/tti_filter_weights.json`.
- Faster runtime: curated NGSIM vehicle traces are cached under `external_data/ngsim/curated/`.
- Better reporting: `experiments/report_breakdown.py` writes real-only, synthetic-only, construction-only, and robustness metrics.
- App evidence panel: the Streamlit app shows trace, geometry, tower, work-zone, and benchmark provenance.
- Backup demo package: `demo_package/` contains the recording checklist and 90-second narration.
- Product framing: `docs/product_path.md` describes cloud, ADAS, and smart-infrastructure deployment paths.
- LNN stability path: training uses multiple real `US-101` traces and includes hard negative edge cases.
- Raw GPS path: Microsoft GeoLife `.plt` ingestion is supported as a separate raw-GPS validation source, distinct from NGSIM.
- Real labeled-event path: partner/DOT wrong-way traces can be imported with `external_data/scripts/import_labeled_trace.py`.

## Quick Start
```bash
bash run.sh
```

The launcher will:
1. install Python dependencies,
2. generate trace data if needed,
3. train cached models if weights are missing,
4. open the Streamlit dashboard on `http://localhost:8501`.

## Experiments
The codebase now includes a small research-style benchmark layout:

```text
experiments/
  exp_1_baseline.py
  exp_2_lnn_ode.py
  exp_3_full_stack.py
  common.py
  results/
```

Each experiment script reproduces one column in the ablation chart:
- `exp_1_baseline.py` -> baseline column
- `exp_2_lnn_ode.py` -> `+ LNN + ODE` column
- `exp_3_full_stack.py` -> full TTI-stack column

The rule baseline itself now lives in [layers/baseline.py](/Users/neha/Downloads/swarmmind/layers/baseline.py), so the repo has an explicit, inspectable answer to the question: "Does the advanced stack beat a simple heading-threshold detector?"

Run them from repo root:

```bash
python3 experiments/exp_1_baseline.py
python3 experiments/exp_2_lnn_ode.py
python3 experiments/exp_3_full_stack.py
```

For repeated training/evaluation rounds of the LNN with real `US-101` traces included, run:

```bash
python3 experiments/calibrate_lnn.py
```

This performs multiple seeded retraining rounds, reruns the benchmark suite each time, and writes an aggregate summary to `/tmp/swarmmind_lnn_calibration.json`.

To calibrate the fusion block and generate the richer report:

```bash
python3 experiments/calibrate_tti_filter.py
python3 experiments/report_breakdown.py
```

The report is saved to:
- `experiments/results/report_breakdown.json`
- `experiments/results/report_breakdown.md`

## Demo Assets
- Pitch kit: [docs/pitch_kit.md](/Users/neha/Downloads/swarmmind/docs/pitch_kit.md)
- Slide outline: [docs/slides_outline.md](/Users/neha/Downloads/swarmmind/docs/slides_outline.md)
- Evidence checklist: [docs/evidence_checklist.md](/Users/neha/Downloads/swarmmind/docs/evidence_checklist.md)
- Product path: [docs/product_path.md](/Users/neha/Downloads/swarmmind/docs/product_path.md)
- Backup package: [demo_package/README.md](/Users/neha/Downloads/swarmmind/demo_package/README.md)

Use the live app’s `construction-zone diversion` trace and the `9-Layer Safety Cascade` flow as the wow-moment sequence. A pre-rendered screencast is still recommended as an off-device backup for judging.
If the staged NGSIM CSV is present, the app also exposes a real `US-101 NGSIM` location plus real-background injected anomaly traces for benchmark and demo use.

## External Validation Plan
The repo now includes an [external_data/README.md](/Users/neha/Downloads/swarmmind/external_data/README.md) plan, a machine-readable [external_data/catalog.json](/Users/neha/Downloads/swarmmind/external_data/catalog.json), and lightweight helper scripts for:
- summarizing the external-data plan,
- fetching OSM hotspot geometry from Overpass,
- normalizing NGSIM CSVs into a SwarmMind-friendly schema,
- pulling OpenCelliD hotspot data once an API key is available.

These are integration-ready scaffolds rather than bundled downloads; the folder is designed so real FHWA/OSM/OpenCelliD assets can be dropped in without restructuring the repo again.

Current workspace note:
- the FHWA `NGSIM` trajectory CSV has been staged under `external_data/ngsim/`,
- curated real trace clips are cached under `external_data/ngsim/curated/`,
- Microsoft GeoLife raw-GPS support is adapter-ready under `external_data/geolife/`,
- real labeled wrong-way event import is adapter-ready under `external_data/labeled_events/`,
- real `OSM/Overpass` corridor extracts have been staged under `external_data/osm/`,
- `OpenCelliD` still needs a user-owned API/download token, so that part remains credential-blocked.
- `WZDx` work-zone ingestion is code-ready; add a direct WZDx GeoJSON URL with `external_data/scripts/fetch_wzdx_feed.py`.

For demo-ready cached traces, run:

```bash
python3 external_data/scripts/extract_ngsim_demo_traces.py
```

## Future Use Cases
1. Wrong-way-driving cloud service for highways.
2. Drunk-driving early warning for ADAS systems.
3. Construction-zone risk monitoring for smart infrastructure.

Under each of these, the reusable core is the same: the TTI stack acts as the anomaly-confirmation module, while different sensing layers can be plugged in later.

## Ethics
The rPPG branch is now visual/demo-only by default and does not change safety confidence in the main pipeline. A production version would need privacy controls, bias mitigation, explicit consent, and secondary sensors before it could influence a real intervention.

## Limitations
- Real wrong-way crash traces are not fabricated. The repo now has a labeled-event ingestion path; claims upgrade only when permitted real traces are staged.
- NGSIM is correctly treated as video-derived real trajectory data, not raw GPS. GeoLife support was added for raw GPS validation, but the dataset must be downloaded separately.
- OpenCelliD requires a user-owned key or extract before Layer 9 can claim real tower density.
- WZDx work-zone truth is integration-ready and registry-aware, but live real-time work-zone claims require a staged WZDx feed.
- A recorded fallback demo is still a manual deliverable.
