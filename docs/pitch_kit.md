# SwarmMind Pitch Kit

## 1-Minute Elevator Pitch
SwarmMind is a 9-layer safety engine for wrong-way detection that turns raw GPS traces into a 15-second warning cascade. The live stack combines adaptive sequence modeling, topology-aware anomaly scoring, continuous-time dynamics, interpretable KAN decisions, and V2X alert routing so we can catch wrong-way vehicles early while suppressing false positives in construction zones and other messy edge cases.

## 3-Minute Core Story
SwarmMind starts with a simple question: is this vehicle moving against the road's expected flow? Layer 1 answers that from GPS and map direction. Layer 3 then separates wrong-way motion from drunk swerving and mechanical drift using the frequency structure of heading changes. On top of that, our TTI stack adds continuous-time learning and anomaly confirmation: the LNN adapts online to the local traffic pattern, the Neural ODE scores how far the trace has moved off the normal driving manifold, persistent topology measures whether the trajectory has entered an abnormal state, and the surprise score turns that anomaly into interpretable bits.

Those signals feed a named fusion block, TTI-Filter, which decides how much to trust the topology score when GPS is noisy. If confidence stays high, KAN exposes the feature-level decision curves, spectral vulnerability tells us how dangerous this road segment is, and Layer 8 fans the alert across nearby vehicles as a ripple. In the demo we show this on Chennai, Los Angeles, and Washington DC with a live map, a layer-status table, and twenty sampled futures converging on the crash zone.

## 5-Minute Deep Dive
LNNs matter here because traffic flow is continuous-time and city-specific; the hidden dynamics can adapt to Chennai rush-hour differently than late-night DC without a fresh retrain.

Persistent topology matters because wrong-way behavior is not just a large heading error; it changes the shape of the trajectory in space-time. That gives us a structural signal instead of one more threshold.

Neural ODEs give us a learned normal-driving flow. When a trace cannot be reconstructed inside that flow, the reconstruction error becomes a clean anomaly scalar.

Surprise bits make that anomaly legible. Normal pings stay low-surprise; impossible states explode in accumulated surprise.

TTI-Filter is our named mechanism: it re-weights topology confidence using ODE error, surprise bits, and trajectory noise so a noisy construction-zone diversion does not look identical to a true wrong-way intrusion.

KAN closes the loop by showing which inputs are pushing the decision upward, and spectral vulnerability lets us say why the same anomaly is more dangerous on one segment than another.

## Demo Theater
Call the live sequence: **The 9-Layer Safety Cascade**.

1. Start with a normal LA trace.
2. Switch to the construction-zone stress preset.
3. Show Layer 1 trigger, TTI-Filter confirmation, and Layer 7 suppression.
4. Jump to the true wrong-way trace so the status table turns fully confirmed and the V2X ripples expand outward.
5. End on RSSM futures and the safe re-route overlay.

## Ethics Line
The rPPG module is demo-only support evidence; any production deployment would require privacy controls, bias audits, consent, and multi-sensor fallback before it could influence a safety-critical decision.
