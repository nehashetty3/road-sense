SwarmMind Safety Net
A 9-Layer Wrong-Way Detection, Collision-Risk Prediction, and V2X Swarm Alert Platform
SwarmMind Safety Net is a production-oriented traffic safety simulation and anomaly-detection platform designed to identify wrong-way vehicles early, suppress false alarms, predict collision zones, and fan out high-confidence alerts across nearby traffic.

It combines a 9-layer safety engine, a research-grade TTI Safety Stack, a live diagnostic simulation cockpit, and a real-data-backed evaluation harness into one unified system.

Why SwarmMind Stands Out
Early wrong-way detection driven by road-direction consensus, continuous-time sequence modeling, and anomaly confirmation.
False-positive suppression for construction detours, bike-adjacent motion, and lane-confusion edge cases.
Collision-risk forecasting with collision cloud overlays and impact-response logic.
V2X-style alert propagation that turns a single anomaly into a surrounding safety response.
Real-data credibility using FHWA NGSIM trajectories plus OSM/Overpass road geometry and real-background stress cases.
Operator-ready simulation UI inspired by advanced automotive diagnostic panels, with live map routing, lane-level scene rendering, config editing, session playback, and alert inspection.
Benchmark Snapshot
Saved benchmark artifacts in experiments/results currently show:

Model	F1 Score	False Positive Rate
Baseline	0.50	0.5714
+ LNN + ODE	0.80	0.1429
Full TTI-Stack	1.00	0.00
Additional saved evaluation highlights:

Real-only full stack: F1 = 1.00, FPR = 0.00
Robustness suite: F1 = 1.00, FPR = 0.00
Wrong-way-only detection: Precision = 1.00, Recall = 1.00, F1 = 1.00
Construction suppression: False Positive Rate = 0.00 for the full stack
These numbers come from the saved benchmark and breakdown artifacts in:

experiments/results/full_stack_pred.json
experiments/results/report_breakdown.json
experiments/results/report_breakdown.md
System Overview
SwarmMind is organized as a modular safety stack:

GPS / Trajectory Stream
-> Road-Direction Consensus
-> LNN Adaptive Sequence Model
-> Neural ODE Reconstruction Error
-> Topological Trajectory Fingerprint
-> Surprise-Bits Anomaly Signal
-> TTI-Filter Fusion
-> KAN Interpretability Layer
-> Spectral Road Vulnerability
-> RSSM Futures + Collision Cloud
-> V2X Ripple Alert Routing
Each stage contributes one interpretable scalar or vector signal, which makes the pipeline robust, composable, and easy to extend with future sensors.

The 9-Layer Safety Engine
Layer 1: GPS Consensus
Detects counter-flow by comparing heading, trajectory direction, and road prior.
Layer 2: Alert Routing
Escalates high-confidence anomalies into private safety alerts and public warnings.
Layer 3: FFT Pattern Analysis
Separates wrong-way behavior from drunk-style oscillation and drift-like faults.
Layer 4: rPPG Demo Signal
Supports richer visualization in the dashboard experience.
Layer 5: Collision Cloud
Predicts impact zones and highlights hot corridors.
Layer 6: Micromobility Guard
Prevents bike-lane or side-path motion from surfacing as road threats.
Layer 7: Work-Zone Guard
Suppresses legal detours and lane diversions in construction-like behavior.
Layer 8: V2X Ripple
Expands alerts outward like a shockwave through nearby vehicles.
Layer 9: Connectivity Assurance
Monitors delivery readiness and supports real-world infrastructure integration.
The TTI Safety Stack
SwarmMind’s advanced stack layers modern anomaly-detection methods on top of the 9-layer engine:

LNN / adaptive temporal model
Learns road-relative sequence behavior and adapts to continuous-time traffic flow.
Neural ODE
Measures how far a trace departs from learned normal motion.
Persistent Homology / TDA
Captures structural differences in trajectory shape over space and time.
Surprise Bits
Converts trajectory improbability into an information-theoretic anomaly score.
TTI-Filter
Reweights anomaly confidence under noisy conditions and stabilizes construction suppression.
KAN
Preserves interpretability through readable feature-level decision structure.
Spectral Vulnerability
Scores road-segment criticality and risk importance.
RSSM Futures
Samples likely forward trajectories to visualize danger convergence.
Real Data + Evaluation Assets
SwarmMind is built around both synthetic scenario control and real-world traffic grounding.

Included data sources
FHWA NGSIM US-101 trajectories
Curated real trace clips cached in external_data/ngsim/curated/us101
OSM / Overpass corridor geometry in external_data/osm
Scenario traces for wrong-way, normal, drunk, fault, bike-path, and construction-detour behavior
Real-background stress cases including short wrong-way, lane confusion, stop-reverse, and construction diversion
Key evaluation outputs
Core ablation benchmarks
Real-only metrics
Construction suppression metrics
Robustness edge-case metrics
Per-case inspection artifacts
Simulation + Operator Experience
SwarmMind ships with a hardened replay and operator workflow:

FastAPI simulation engine in simulation_engine.py
Session-based backend service in simulation_backend.py
Toolbox3-inspired simulation cockpit served directly in the browser
Streamlit control panel in ui_toolbox3_panel.py
Config versioning, rollback, session controls, metrics, audit logs
Live route pane with Mapbox support
Collision-response visualization, lane-level rendering, and alert timeline
Quick Start
Run the main dashboard experience:

bash run.sh
Or start the simulation engine directly:

uvicorn simulation_engine:app --host 127.0.0.1 --port 8000 --reload
Then open:

http://127.0.0.1:8000/
For the Streamlit panel:

streamlit run ui_toolbox3_panel.py --server.port 8502
Experiments
Research-style experiment entrypoints live in experiments:

experiments/
  exp_1_baseline.py
  exp_2_lnn_ode.py
  exp_3_full_stack.py
  calibrate_lnn.py
  calibrate_tti_filter.py
  report_breakdown.py
Run them from repo root:

python3 experiments/exp_1_baseline.py
python3 experiments/exp_2_lnn_ode.py
python3 experiments/exp_3_full_stack.py
python3 experiments/calibrate_lnn.py
python3 experiments/calibrate_tti_filter.py
python3 experiments/report_breakdown.py
Demo Story
The strongest live demo flow is:

Start with a normal divided-highway trace
Inject a wrong-way event
Show the 9-layer table activate
Visualize the collision cloud and impact response
Watch V2X ripples spread through the traffic scene
Compare with construction detour and bike-path suppression
Supporting assets:

docs/pitch_kit.md
docs/slides_outline.md
docs/evidence_checklist.md
demo_package/README.md
Deployment
The repo is ready for cloud deployment:

GitHub-ready
Render-ready via render.yaml
Mapbox-ready for live route maps using SWARMMIND_MAPBOX_PUBLIC_TOKEN
To enable the live map route pane:

SWARMMIND_MAPBOX_PUBLIC_TOKEN=pk.your_public_mapbox_token
SWARMMIND_MAPBOX_STYLE=mapbox://styles/mapbox/dark-v11
Project Structure
SwarmMind/
├── app.py
├── simulation_engine.py
├── simulation_backend.py
├── ui_toolbox3_panel.py
├── layers/
├── experiments/
├── data/
├── external_data/
├── docs/
├── demo_package/
└── tests/
Future Integrations
SwarmMind is designed to expand cleanly into richer sensing and infrastructure workflows:

radar track ingestion
LiDAR point-cloud overlays
richer V2X CAM / RSU inputs
live work-zone feeds
real cellular coverage overlays
additional ADAS and smart-infrastructure interfaces
One-Line Pitch
SwarmMind Safety Net is a modular, high-confidence wrong-way detection and traffic safety platform that fuses real trajectory data, advanced anomaly modeling, collision forecasting, and V2X-style alert propagation into a live operator-ready safety system.
