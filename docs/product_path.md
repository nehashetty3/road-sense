# SwarmMind Product Path

## Near-Term
- Highway cloud service: ingest GPS or V2X CAM streams, run the TTI stack, and push corridor-level wrong-way warnings.
- ADAS early warning: use SwarmMind as a safety-prior module that warns before onboard perception sees the vehicle.
- Smart-infrastructure dashboard: combine OSM geometry, WZDx work zones, and tower density to identify fragile road segments.

## Integrations
- Radar tracks can replace or augment GPS heading.
- LiDAR point clouds can become an additional trajectory feature block.
- V2X CAM streams can feed the same `GPS -> LNN -> ODE -> TDA -> Surprise -> KAN -> V2X` pipeline.

## Deployment Guardrails
- rPPG remains optional and non-blocking.
- Alerts should require at least two independent confirmation signals.
- Work-zone and emergency-vehicle feeds should suppress public alerts until confidence clears the TTI threshold.
