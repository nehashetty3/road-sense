# rPPG Bias and Privacy Notes

The rPPG signal in SwarmMind is a demo-only confidence feature, not a required safety signal.

Production use would need:
- explicit consent before biometric processing,
- skin-tone and lighting bias evaluation,
- on-device processing or privacy-preserving aggregation,
- non-biometric fallback sensors,
- and a policy that prevents rPPG from being the sole trigger for an intervention.

Judge-facing line:
"The rPPG module is optional support evidence; a production deployment would require privacy controls, bias mitigation, consent, and multi-sensor fallbacks."
