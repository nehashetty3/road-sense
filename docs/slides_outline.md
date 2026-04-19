# Slide-Ready Outline

## Pipeline Graph
GPS / map direction -> F1 LNN -> F6 Neural ODE -> F2 TDA -> F9 Surprise Bits -> TTI-Filter -> F4 KAN -> F5 Spectral Vulnerability -> L8/L9 V2X

Presenter note:
Each block outputs one scalar or vector, and the blocks are decoupled enough that a failure in any single advanced module does not collapse the entire safety stack.

## Literature Anchors
- LNNs for traffic adaptation: continuous-time recurrent modeling for streaming mobility signals.
- TDA for trajectory anomaly detection: persistence features over space-time traces.
- Neural ODEs for vehicle dynamics: continuous latent flows for learned trajectory reconstruction.
- KANs for interpretability: curve-based neural decisions that remain auditable.
- KLD / surprise-based anomaly scoring: information-theoretic outlier quantification.
- Spectral vulnerability on road graphs: graph connectivity and segment criticality analysis.

Positioning line:
SwarmMind composes these methods into a new TTI-Safety-Stack for wrong-way detection, false-positive suppression, and downstream V2X alerting.

## Future Use Cases
1. Wrong-way-driving cloud service for highways.
2. Drunk-driving early warning for ADAS systems.
3. Construction-zone risk monitoring for smart infrastructure.

Product path:
Future integrations can plug radar tracks, LiDAR point clouds, or V2X CAM streams into the same TTI pipeline.
