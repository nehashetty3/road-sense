#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SwarmMind – one-command launcher
# Usage: bash run.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  🛡️  SwarmMind – Wrong-Way Safety Engine             ║"
echo "  ║  9-Layer Core  +  9 Advanced ML Features             ║"
echo "  ║  MIT-Mahe × HARMAN Automotive Hackathon              ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""

# Install dependencies (idempotent)
echo "  [1/3] Checking dependencies…"
pip install -r requirements.txt --break-system-packages -q 2>/dev/null || \
pip install -r requirements.txt -q 2>/dev/null || true

# Generate data if missing
if [ ! -f "data/wrongway_chennai.csv" ]; then
    echo "  [2/3] Generating synthetic dataset…"
    python generate_data.py
else
    echo "  [2/3] Dataset found (skipping generation)"
fi

# Train models if missing
MODELS_NEEDED=("data/lnn_weights.pt" "data/nca_weights.pt" "data/ode_weights.pt"
               "data/hyper_weights.pt" "data/rssm_weights.pt" "data/kan_weights.pt")
NEED_TRAIN=0
for f in "${MODELS_NEEDED[@]}"; do
    [ ! -f "$f" ] && NEED_TRAIN=1 && break
done
if [ $NEED_TRAIN -eq 1 ]; then
    echo "  [3/3] Training ML models (takes ~30s)…"
    python train_models.py
else
    echo "  [3/3] Trained models found (skipping training)"
fi

echo ""
echo "  ✓ Ready!  Opening SwarmMind at http://localhost:8501"
echo ""
streamlit run app.py \
    --server.port 8501 \
    --server.headless false \
    --theme.base dark \
    --theme.primaryColor "#ff2d55" \
    --theme.backgroundColor "#0d1117" \
    --theme.secondaryBackgroundColor "#161b22" \
    --theme.textColor "#c9d1d9"
