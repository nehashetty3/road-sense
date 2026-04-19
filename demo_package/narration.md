# 90-Second Demo Narration

SwarmMind starts with a real FHWA NGSIM freeway trace on US-101. On normal traffic, the baseline and the TTI stack stay quiet.

Now we inject a construction-zone diversion on that same real background. A simple heading rule panics, but Layer 7 and TTI-Filter suppress the false positive because the motion is noisy, short-lived, and consistent with a detour.

Finally, we inject a sustained wrong-way intrusion. Layer 1 sees counter-flow, the LNN and ODE score the trace as abnormal, topology and surprise bits confirm it, KAN keeps the decision explainable, and Layer 8 ripples the alert across nearby vehicles.

The result is the 9-Layer Safety Cascade: fewer false alarms, real-data grounding, and a warning that can arrive before the driver only has a few seconds left.
