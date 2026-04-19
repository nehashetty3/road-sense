# SwarmMind Safety Net · Toolbox3-Inspired Simulation

This adds a hardened simulation API and a Tesla-Toolbox3-inspired diagnostic UI for replaying wrong-way traces, inspecting the 9-layer stack, editing thresholds, and watching V2X alert ripples.

## Files
- `simulation_engine.py`: FastAPI backend with replay, session control, config management, metrics, and diagnostic routes.
- `ui_toolbox3_panel.py`: Streamlit diagnostic panel that controls the backend.
- `csv_demo/`: auto-created demo CSV folder using existing SwarmMind traces.
- `simulation_backend.py`: persisted session service, SQLite store, and SwarmMind runtime adapter.

## Start the API
From the SwarmMind repo root:

```bash
uvicorn simulation_engine:app --host 127.0.0.1 --port 8000 --reload
```

Useful endpoints:

```text
GET  /health
GET  /state
GET  /state/{session_id}
GET  /config
POST /config
GET  /config/history
POST /config/rollback
GET  /presets
POST /presets
GET  /csvs
GET  /sessions
POST /sessions/{session_id}/control
GET  /alerts/{session_id}
GET  /events/{session_id}
GET  /logs
GET  /metrics
POST /maintenance/cleanup
POST /playback
GET  /ws/{session_id}
```

Example playback request:

```bash
curl -N -X POST http://127.0.0.1:8000/playback \
  -H "Content-Type: application/json" \
  -d '{"csv_filename":"wrong_way.csv","delay_ms":35,"max_seconds":90}'
```

Each streamed event is SSE-style:

```text
data: {"time": 10.0, "vehicles": [...], "alert": {...}, "layers": [...]}
```

## Start the UI
In a second terminal:

```bash
streamlit run ui_toolbox3_panel.py --server.port 8502
```

Then open:

```text
http://127.0.0.1:8502
```

If the API is running somewhere else:

```bash
SWARMMIND_API_URL=http://127.0.0.1:8000 streamlit run ui_toolbox3_panel.py --server.port 8502
```

## Run the 90-Second Demo Trace
1. Start the API.
2. Start the UI.
3. Select `wrong_way.csv`.
4. Keep demo length at `90`.
5. Press `Play Trace`.

The UI will stream frames into:
- real-time status bar,
- central dark map,
- 9-layer inspector,
- V2X ripple overlays,
- alert timeline,
- raw `/state` JSON.

## Determinism
The replay is deterministic:
- CSV path resolution is stable.
- Road direction is inferred from the trace name.
- The simulation engine uses deterministic formulas and a filename-derived seed.
- Same CSV + same config gives the same streamed event sequence.

## Config Controls
The UI writes to `POST /config`:
- `lnn_threshold`
- `ode_threshold`
- `tti_threshold`
- `wrong_way_delta`
- `min_wrong_steps`
- `kan_weights`

The API keeps config in memory for simulation sessions. This makes it safe for demos and easy to plug into the existing Streamlit dashboard later.

## Operational Notes
- Sessions are persisted in `simulation_state.db` by default.
- Every HTTP response includes an `x-request-id` header for traceability.
- Config changes are versioned, rollback-capable, and can be saved as named presets.
- Session playback supports `pause`, `resume`, and `cancel` through `POST /sessions/{session_id}/control`.
- Metrics and audit logs are available through `/metrics` and `/logs`.
- Runtime dependency status is exposed through `/readiness` and `/dependencies`.
- Environment-driven startup settings live in `.env.example`.
- Optional request auth is available with `SWARMMIND_AUTH_MODE=api_key` or `SWARMMIND_AUTH_MODE=bearer`.
- Bearer mode now verifies signed JWTs with `SWARMMIND_BEARER_SECRET`.
- `SWARMMIND_STRICT_STARTUP=true` can refuse startup when production-like settings are unsafe.
- The route pane can use a live Mapbox map when `SWARMMIND_MAPBOX_PUBLIC_TOKEN` is set; otherwise it falls back to the built-in vector map.

## Live Map Option
SwarmMind now supports Mapbox GL JS in the route overview pane. This is the recommended map provider for the project because it has a browser-friendly public-token model and a real free web-map tier.

Add these values to `.env`:

```bash
SWARMMIND_MAPBOX_PUBLIC_TOKEN=pk.your_public_mapbox_token
SWARMMIND_MAPBOX_STYLE=mapbox://styles/mapbox/dark-v11
```

If no token is set, the dashboard still works and uses the local stylized route map instead.

## GitHub + Render Deploy
This repo now includes [render.yaml](/Users/neha/Downloads/swarmmind/render.yaml) for a free Render web-service deployment.

Render settings used:
- Runtime: `Python`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn simulation_engine:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

After you push the repo to GitHub:
1. Create a free Render account.
2. Choose `New > Web Service`.
3. Connect your GitHub account and select the SwarmMind repo.
4. Render will detect `render.yaml` automatically.
5. Add `SWARMMIND_MAPBOX_PUBLIC_TOKEN` in Render if you want the live map enabled.

Important free-tier note from Render’s docs: free web services can spin down after inactivity and are not appropriate for true production workloads.

## Current Infra Boundary
The backend is now materially harder and more operationally usable, but fully production-grade deployment still needs external infrastructure that is outside local repo code:
- PostgreSQL or another multi-client DB if you want strong concurrent production usage beyond SQLite.
- Real authentication / authorization if the API will be exposed outside localhost.
- Real external feeds such as OpenCelliD, WZDx, or OEM/DOT event sources if you want live infrastructure-backed simulation inputs.

## Suggested Local Secure Run
```bash
export SWARMMIND_AUTH_MODE=api_key
export SWARMMIND_API_KEY=change-this
export SWARMMIND_ALLOW_INSECURE_LOCALHOST=false
uvicorn simulation_engine:app --host 127.0.0.1 --port 8000 --reload
```

Then call protected endpoints with:

```bash
curl -H "x-api-key: change-this" http://127.0.0.1:8000/state
```

## Suggested Production-Like Run
```bash
export SWARMMIND_ENV=production
export SWARMMIND_STRICT_STARTUP=true
export SWARMMIND_DATABASE_URL=postgresql://user:pass@host:5432/swarmmind
export SWARMMIND_AUTH_MODE=bearer
export SWARMMIND_BEARER_SECRET=replace-with-a-strong-secret
export SWARMMIND_BEARER_AUDIENCE=swarmmind-api
export SWARMMIND_BEARER_ISSUER=https://swarmmind.local
uvicorn simulation_engine:app --host 0.0.0.0 --port 8000
```
