# SwarmMind Demo Package

This folder is the offline backup plan for judging. The app can be live, but the pitch should not depend on Wi-Fi, Streamlit, or a webcam.

## Required Files
- `swarmmind_90s_demo.mp4`: screen recording of the full flow.
- `screenshots/01_source_panel.png`: evidence panel showing FHWA NGSIM + OSM.
- `screenshots/02_ablation.png`: baseline vs LNN+ODE vs full TTI-stack.
- `screenshots/03_safety_cascade.png`: Layer 1 trigger, TTI confirmation, V2X ripple.
- `screenshots/04_rssm_futures.png`: sampled future paths and safe-route overlay.

## Recording Sequence
1. Open the Streamlit dashboard.
2. Select `US-101 NGSIM, Los Angeles`.
3. Run `Real NGSIM normal`.
4. Switch to `Real NGSIM + construction injection` and show baseline alarm vs stack suppression.
5. Switch to `Real NGSIM + wrong-way injection`.
6. Show the 9-layer safety cascade, V2X ripples, ablation chart, and evidence panel.

## Backup Line
If the live app fails, say: "This is the exact workflow running locally; the video is a pre-recorded run of the same repo and the same saved benchmark artifacts."
