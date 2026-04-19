# SwarmMind Evaluation Breakdown

## Full TTI-Stack
- Validation tier: prototype_demo_validation
- Overall: F1=1.0000, FPR=0.0000
- Real-only core: F1=1.0000, FPR=0.0000
- Robustness suite: F1=1.0000, FPR=0.0000

## Source Provenance
- Trace source: FHWA NGSIM
- Raw GPS source: not staged
- Real labeled events: not staged
- Geometry source: OSM/Overpass
- Connectivity source: synthetic tower map
- Work-zone source: synthetic/no feed
- Curated real traces: 4

## Notes
- Core ablation metrics come from the saved benchmark JSON files.
- Robustness cases extend the real US-101 trace with short wrong-way, lane-confusion, and stop-reverse edge cases.
- Scores are prototype validation until raw GPS, real labeled events, OpenCelliD, and live WZDx assets are staged.
- OpenCelliD remains credential-gated until user credentials are supplied.
