from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import math
import os
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api_security import enforce_request_auth
from infra_status import readiness_report, startup_validation_errors
from layers.core import _heading_delta
from production_settings import load_settings
from simulation_backend import ConfigUpdate as BackendConfigUpdate
from simulation_backend import PlaybackRequest as BackendPlaybackRequest
from simulation_backend import InvalidConfigError, SessionNotFoundError, SimulationError
from simulation_backend import SimulationService


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_DEMO_DIR = ROOT / "csv_demo"
SETTINGS = load_settings()
DB_PATH = SETTINGS.database_target
MAPBOX_PUBLIC_TOKEN = SETTINGS.mapbox_public_token or ""
MAPBOX_STYLE = SETTINGS.mapbox_style

DEFAULT_CONFIG = {
    "lnn_threshold": 0.30,
    "ode_threshold": 0.05,
    "tti_threshold": 0.55,
    "wrong_way_delta": 90.0,
    "min_wrong_steps": 8,
    "v2x_radius_km": 3.0,
    "kan_weights": {
        "heading_delta": 0.42,
        "sustained_wrongway": 0.24,
        "speed": 0.12,
        "construction_penalty": -0.18,
        "connectivity": 0.04,
    },
}

CSV_ALIASES = {
    "normal.csv": "normal_la.csv",
    "wrong_way.csv": "wrongway_la.csv",
    "wrong_way_la.csv": "wrongway_la.csv",
    "construction_detour.csv": "construction_la.csv",
    "bike_path.csv": "normal_la.csv",
}


class PlaybackRequest(BaseModel):
    csv_filename: str = Field(..., examples=["wrong_way.csv"])
    delay_ms: int = Field(40, ge=0, le=1000)
    max_seconds: int | None = Field(90, ge=1, le=3600)


class ConfigUpdate(BaseModel):
    lnn_threshold: float | None = Field(None, ge=0.0, le=1.0)
    ode_threshold: float | None = Field(None, ge=0.0)
    tti_threshold: float | None = Field(None, ge=0.0, le=1.0)
    wrong_way_delta: float | None = Field(None, ge=0.0, le=180.0)
    min_wrong_steps: int | None = Field(None, ge=1, le=500)
    v2x_radius_km: float | None = Field(None, ge=0.1, le=20.0)
    kan_weights: dict[str, float] | None = None


class PresetRequest(BaseModel):
    preset_name: str = Field(..., min_length=1, max_length=64)


class ConfigRollbackRequest(BaseModel):
    version: int = Field(..., ge=1)


class SessionControlRequest(BaseModel):
    action: str = Field(..., pattern="^(pause|resume|cancel)$")


class TracePreviewRequest(BaseModel):
    csv_filename: str = Field(..., examples=["wrong_way.csv"])
    max_seconds: int | None = Field(90, ge=1, le=3600)


def to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, SessionNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (InvalidConfigError, SimulationError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@dataclass
class SimulationState:
    time: float = 0.0
    status: str = "NORMAL"
    vehicles: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    layers: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))
    active_csv: str | None = None
    playback_running: bool = False
    frame_index: int = 0


STATE = SimulationState()
STATE_LOCK = threading.Lock()


def ensure_csv_demo() -> None:
    CSV_DEMO_DIR.mkdir(exist_ok=True)
    copies = {
        "normal.csv": "normal_la.csv",
        "wrong_way.csv": "wrongway_la.csv",
        "wrong_way_la.csv": "wrongway_la.csv",
        "bike_path.csv": "normal_la.csv",
    }
    for target_name, source_name in copies.items():
        target = CSV_DEMO_DIR / target_name
        source = DATA_DIR / source_name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)

    construction_target = CSV_DEMO_DIR / "construction_detour.csv"
    if not construction_target.exists():
        source = DATA_DIR / "construction_la.csv"
        if source.exists():
            shutil.copy2(source, construction_target)
        elif (DATA_DIR / "normal_la.csv").exists():
            df = pd.read_csv(DATA_DIR / "normal_la.csv")
            idx = (df["time"] >= 30) & (df["time"] <= 75)
            df.loc[idx, "heading"] = (df.loc[idx, "heading"] + np.linspace(15, 42, int(idx.sum()))) % 360
            df.loc[idx, "speed"] = np.clip(df.loc[idx, "speed"] * 0.72, 8.0, None)
            df.to_csv(construction_target, index=False)


def resolve_csv_path(csv_filename: str) -> Path:
    ensure_csv_demo()
    clean_name = Path(csv_filename).name
    candidates = [
        CSV_DEMO_DIR / clean_name,
        CSV_DEMO_DIR / CSV_ALIASES.get(clean_name, clean_name),
        DATA_DIR / clean_name,
        DATA_DIR / CSV_ALIASES.get(clean_name, clean_name),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix == ".csv":
            return candidate
    raise HTTPException(status_code=404, detail=f"CSV not found: {csv_filename}")


def infer_road_direction(csv_name: str) -> float:
    name = csv_name.lower()
    if "dc" in name:
        return 45.0
    if "chennai" in name:
        return 180.0
    return 350.0


def deterministic_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class SwarmMindEngine9Layer:
    """Deterministic adapter around SwarmMind's 9-layer safety logic."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = json.loads(json.dumps(config or DEFAULT_CONFIG))
        self.wrong_streak = 0
        self.alert_log: list[dict[str, Any]] = []

    def reset(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            self.config = json.loads(json.dumps(config))
        self.wrong_streak = 0
        self.alert_log = []

    def process_timestep(self, row: pd.Series, road_direction: float, csv_name: str) -> dict[str, Any]:
        heading = float(row["heading"])
        speed = float(row["speed"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        t = float(row["time"])

        road_delta = _heading_delta(heading, road_direction)
        wrong = road_delta >= float(self.config["wrong_way_delta"])
        self.wrong_streak = self.wrong_streak + 1 if wrong else 0

        lnn_score = min(1.0, max(0.0, (road_delta - 35.0) / 145.0))
        ode_error = min(1.0, max(0.0, abs(speed - 45.0) / 80.0 + max(0.0, road_delta - 70.0) / 240.0))
        kan_score = self._kan_score(road_delta, speed, csv_name)
        tti_confidence = min(1.0, 0.45 * lnn_score + 0.35 * ode_error + 0.20 * kan_score)

        confirmed = (
            self.wrong_streak >= int(self.config["min_wrong_steps"])
            and lnn_score >= float(self.config["lnn_threshold"])
            and ode_error >= min(float(self.config["ode_threshold"]), 0.99)
            and tti_confidence >= float(self.config["tti_threshold"])
        )

        alert = None
        if confirmed:
            alert = {
                "time": round(t, 2),
                "type": "Wrong-Way-Detected",
                "lat": lat,
                "lon": lon,
                "confidence": round(tti_confidence, 3),
                "location": f"{lat:.5f}, {lon:.5f}",
                "v2x_radius_km": self.config["v2x_radius_km"],
            }
            if not self.alert_log or self.alert_log[-1].get("time") != alert["time"]:
                self.alert_log.append(alert)

        behavior = self._behavior_label(csv_name, confirmed, wrong)
        vehicles = self._build_scene_vehicles(
            csv_name=csv_name,
            t=t,
            lat=lat,
            lon=lon,
            heading=heading,
            speed=speed,
            behavior=behavior,
            confirmed=confirmed,
            wrong=wrong,
        )
        layers = self.compute_9_layer_status(
            road_delta=road_delta,
            lnn_score=lnn_score,
            ode_error=ode_error,
            kan_score=kan_score,
            tti_confidence=tti_confidence,
            confirmed=confirmed,
            csv_name=csv_name,
        )
        return {
            "time": round(t, 2),
            "vehicles": vehicles,
            "alert": alert,
            "layers": layers,
            "status": "WRONG-WAY DETECTED" if confirmed else "CAUTION" if wrong else "NORMAL",
        }

    def compute_9_layer_status(
        self,
        road_delta: float,
        lnn_score: float,
        ode_error: float,
        kan_score: float,
        tti_confidence: float,
        confirmed: bool,
        csv_name: str,
    ) -> list[dict[str, Any]]:
        construction_like = "construction" in csv_name.lower() or "detour" in csv_name.lower()
        suppressed = construction_like and not confirmed
        rows = [
            ("L1 GPS Consensus", min(1.0, road_delta / 180.0), "active" if road_delta > 45 else "idle"),
            ("L2 Alert Routing", tti_confidence, "active" if confirmed else "idle"),
            ("L3 FFT Pattern", lnn_score, "active" if lnn_score > 0.45 else "idle"),
            ("L4 rPPG Visual Only", 0.0, "suppressed"),
            ("L5 Collision Cloud", tti_confidence, "active" if confirmed else "idle"),
            ("L6 Micromobility Guard", 0.7 if "bike" in csv_name.lower() else 0.2, "suppressed" if "bike" in csv_name.lower() else "idle"),
            ("L7 Work-Zone Guard", 0.8 if construction_like else 0.2, "suppressed" if suppressed else "active" if construction_like else "idle"),
            ("L8 V2X Ripple", 0.95 if confirmed else 0.1, "active" if confirmed else "idle"),
            ("L9 Connectivity", 0.78, "active"),
        ]
        return [
            {
                "layer": layer,
                "confidence": round(float(conf), 3),
                "status": "failed" if status == "active" and conf < 0.15 else status,
            }
            for layer, conf, status in rows
        ]

    def _kan_score(self, road_delta: float, speed: float, csv_name: str) -> float:
        weights = self.config.get("kan_weights", {})
        construction = 1.0 if "construction" in csv_name.lower() or "detour" in csv_name.lower() else 0.0
        score = (
            float(weights.get("heading_delta", 0.42)) * (road_delta / 180.0)
            + float(weights.get("sustained_wrongway", 0.24)) * min(1.0, self.wrong_streak / max(1, self.config["min_wrong_steps"]))
            + float(weights.get("speed", 0.12)) * min(1.0, speed / 120.0)
            + float(weights.get("construction_penalty", -0.18)) * construction
            + float(weights.get("connectivity", 0.04)) * 0.78
        )
        return 1.0 / (1.0 + math.exp(-4.0 * (score - 0.35)))

    def _build_scene_vehicles(
        self,
        csv_name: str,
        t: float,
        lat: float,
        lon: float,
        heading: float,
        speed: float,
        behavior: str,
        confirmed: bool,
        wrong: bool,
    ) -> list[dict[str, Any]]:
        lower = csv_name.lower()
        lane_centers = {0: 28.0, 1: 42.0, 2: 58.0, 3: 72.0}
        ego_wrong = "wrong" in lower
        construction_like = "construction" in lower or "detour" in lower
        bike_like = "bike" in lower

        def normal_progress(seed: float, velocity: float) -> float:
            return (seed + (t * velocity)) % 118.0 - 9.0

        def inverted_progress(seed: float, velocity: float) -> float:
            return 100.0 - ((seed + (t * velocity)) % 118.0 - 9.0)

        ego_lane = 3 if ego_wrong else 1
        if construction_like:
            ego_lane = 2 if int(t // 8) % 2 == 0 else 1
        ego_progress = inverted_progress(12.0, max(0.35, speed / 115.0)) if ego_wrong else normal_progress(18.0, max(0.35, speed / 130.0))
        vehicles = [
            {
                "id": "ego-001",
                "lat": lat,
                "lon": lon,
                "heading": round(heading, 2),
                "speed": round(speed, 2),
                "behavior": behavior,
                "lane": ego_lane,
                "progress": round(float(ego_progress), 2),
                "direction": "southbound" if ego_wrong else "northbound",
                "is_alert_source": confirmed or wrong,
                "sprite": "sedan",
            }
        ]

        traffic_specs = [
            ("north-01", 0, 8.0, 0.60, "normal", "northbound", "sedan"),
            ("north-02", 1, 54.0, 0.55, "normal", "northbound", "suv"),
            ("north-03", 0, 84.0, 0.48, "normal", "northbound", "sedan"),
            ("south-01", 2, 22.0, 0.52, "normal", "southbound", "suv"),
            ("south-02", 3, 66.0, 0.58, "normal", "southbound", "sedan"),
            ("south-03", 2, 96.0, 0.44, "normal", "southbound", "truck"),
        ]

        for vid, lane, seed, velocity, base_behavior, direction, sprite in traffic_specs:
            lane_override = lane
            sim_behavior = base_behavior
            if construction_like and direction == "northbound" and lane == 1 and 25.0 < t < 58.0:
                lane_override = 0
                sim_behavior = "detour"
            if bike_like and vid == "north-03":
                lane_override = 0
                sim_behavior = "bike-path-nearby"
            progress = normal_progress(seed, velocity) if direction == "northbound" else inverted_progress(seed, velocity)
            vehicles.append(
                {
                    "id": vid,
                    "lat": round(lat + (lane_centers[lane_override] - 50.0) * 0.00001, 6),
                    "lon": round(lon + (progress - 50.0) * 0.00001, 6),
                    "heading": 0.0 if direction == "northbound" else 180.0,
                    "speed": round(38.0 + velocity * 28.0, 2),
                    "behavior": sim_behavior,
                    "lane": lane_override,
                    "progress": round(float(progress), 2),
                    "direction": direction,
                    "is_alert_source": False,
                    "sprite": sprite,
                }
            )
        return vehicles

    @staticmethod
    def _behavior_label(csv_name: str, confirmed: bool, wrong: bool) -> str:
        lower = csv_name.lower()
        if confirmed:
            return "wrong-way-confirmed"
        if "construction" in lower or "detour" in lower:
            return "construction-detour"
        if "bike" in lower:
            return "bike-path-nearby"
        if wrong:
            return "counter-flow-caution"
        return "normal"


def run_simulation_from_csv(filename: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    path = resolve_csv_path(filename)
    df = pd.read_csv(path).sort_values("time").reset_index(drop=True)
    road_direction = infer_road_direction(path.name)
    np.random.seed(deterministic_seed(path.name))
    engine = SwarmMindEngine9Layer(config or STATE.config)
    events = []
    for _, row in df.iterrows():
        events.append(engine.process_timestep(row, road_direction, path.name))
    return events


def current_state_dict() -> dict[str, Any]:
    with STATE_LOCK:
        return asdict(STATE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    errors = startup_validation_errors(SETTINGS, ROOT)
    if errors and SETTINGS.strict_startup:
        raise RuntimeError("Startup validation failed: " + " | ".join(errors))
    SERVICE.bootstrap_state()
    yield


app = FastAPI(title="SwarmMind Safety Net Simulation API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
SERVICE = SimulationService(DB_PATH)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    try:
        enforce_request_auth(request, SETTINGS)
        response = await call_next(request)
    except HTTPException as exc:
        response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    response.headers["x-request-id"] = request_id
    SERVICE.db.add_request_log(request.method, request.url.path, response.status_code, request_id=request_id)
    return response


ROOT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SwarmMind Safety Net</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.20.0/mapbox-gl.css" rel="stylesheet" />
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.20.0/mapbox-gl.js"></script>
  <style>
    :root {
      --bg: #070a0f;
      --panel: #0d141f;
      --panel2: #111b29;
      --line: #233247;
      --text: #e7edf7;
      --muted: #7f8da3;
      --green: #29f19c;
      --yellow: #ffd166;
      --red: #ff3b5f;
      --blue: #59a6ff;
      --orange: #ff8a3d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 20% 0%, rgba(89,166,255,.12), transparent 32%),
        radial-gradient(circle at 80% 20%, rgba(255,59,95,.11), transparent 30%),
        var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(7,10,15,.82);
      backdrop-filter: blur(18px);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 { font-size: 21px; margin: 0; letter-spacing: .08em; text-transform: uppercase; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr 390px;
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 72px);
    }
    .panel {
      background: linear-gradient(180deg, rgba(17,27,41,.94), rgba(11,17,26,.94));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 18px 70px rgba(0,0,0,.35);
    }
    .panel h2 { margin: 0 0 12px; font-size: 13px; color: #cbd6e8; letter-spacing: .1em; text-transform: uppercase; }
    .status {
      border-radius: 16px;
      padding: 18px;
      text-align: center;
      font-weight: 900;
      letter-spacing: .08em;
      text-transform: uppercase;
      border: 1px solid var(--line);
      margin-bottom: 14px;
    }
    .NORMAL { background: rgba(41,241,156,.12); color: var(--green); border-color: rgba(41,241,156,.38); }
    .CAUTION { background: rgba(255,209,102,.13); color: var(--yellow); border-color: rgba(255,209,102,.38); }
    .WRONG { background: rgba(255,59,95,.16); color: var(--red); border-color: rgba(255,59,95,.45); animation: pulse 1s infinite alternate; }
    @keyframes pulse { from { box-shadow: 0 0 0 rgba(255,59,95,0); } to { box-shadow: 0 0 28px rgba(255,59,95,.35); } }
    label { color: var(--muted); font-size: 12px; display: block; margin: 12px 0 6px; }
    select, input {
      width: 100%;
      background: #070c13;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 11px;
      outline: none;
    }
    button {
      border: 0;
      border-radius: 13px;
      padding: 12px 14px;
      color: #06100b;
      background: linear-gradient(135deg, var(--green), #83ffd0);
      font-weight: 900;
      width: 100%;
      cursor: pointer;
      margin-top: 14px;
    }
    button.secondary { background: #172235; color: var(--text); border: 1px solid var(--line); }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 13px; }
    .metric .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .metric .v { font-size: 25px; font-weight: 900; margin-top: 4px; }
    .map {
      height: 560px;
      border-radius: 24px;
      border: 1px solid var(--line);
      overflow: hidden;
      position: relative;
      display: grid;
      grid-template-columns: 31% 69%;
      background: linear-gradient(180deg, #07090d, #040507 72%);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.04), 0 30px 90px rgba(0,0,0,.35);
    }
    .nav-pane {
      position: relative;
      background:
        radial-gradient(circle at 18% 22%, rgba(89,166,255,.09), transparent 18%),
        linear-gradient(180deg, rgba(10,13,19,.95), rgba(6,8,12,.98));
      border-right: 1px solid rgba(255,255,255,.05);
      padding: 18px 16px 18px 18px;
    }
    .nav-pane::after {
      content: "";
      position: absolute;
      inset: 14px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,.04);
      pointer-events: none;
    }
    .nav-label {
      font-size: 11px;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .nav-route {
      position: relative;
      height: 420px;
      border-radius: 18px;
      overflow: hidden;
      background:
        radial-gradient(circle at 60% 25%, rgba(66,116,255,.12), transparent 20%),
        linear-gradient(180deg, #090b11, #05070a);
      border: 1px solid rgba(255,255,255,.06);
    }
    .nav-route.has-map .nav-grid,
    .nav-route.has-map svg {
      opacity: 0;
      pointer-events: none;
    }
    #mapboxRouteMap {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: none;
    }
    .nav-route.has-map #mapboxRouteMap {
      display: block;
    }
    .mapboxgl-map,
    .mapboxgl-canvas {
      border-radius: 18px;
    }
    .mapboxgl-ctrl-bottom-left,
    .mapboxgl-ctrl-bottom-right {
      transform: scale(.82);
      transform-origin: bottom right;
    }
    .mapbox-status {
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 2;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(5,8,12,.84);
      color: #bfd0ef;
      border: 1px solid rgba(255,255,255,.08);
      font-size: 10px;
      letter-spacing: .08em;
      text-transform: uppercase;
      backdrop-filter: blur(10px);
    }
    .mapbox-status.live {
      color: #9bd2ff;
    }
    .nav-grid {
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px);
      background-size: 26px 26px;
      opacity: .18;
    }
    .nav-route svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    .nav-map-minor {
      fill: none;
      stroke: rgba(154, 173, 204, 0.12);
      stroke-width: 5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .nav-map-road {
      fill: none;
      stroke: rgba(198, 212, 236, 0.2);
      stroke-width: 10;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .nav-map-route-shadow,
    .nav-map-route,
    .nav-map-detour,
    .nav-map-bike {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .nav-map-route-shadow {
      stroke: rgba(27, 87, 216, 0.34);
      stroke-width: 15;
    }
    .nav-map-route {
      stroke: #2f8fff;
      stroke-width: 7;
    }
    .nav-map-detour {
      stroke: #ffbf5a;
      stroke-width: 6;
      stroke-dasharray: 10 8;
      opacity: 0;
      transition: opacity .2s ease;
    }
    .nav-map-bike {
      stroke: #52f0ae;
      stroke-width: 5;
      stroke-dasharray: 6 6;
      opacity: 0;
      transition: opacity .2s ease;
    }
    .nav-pane.construction .nav-map-detour { opacity: 1; }
    .nav-pane.bike .nav-map-bike { opacity: 1; }
    .nav-map-label {
      fill: rgba(226, 235, 251, 0.76);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .02em;
    }
    .nav-map-shield rect {
      fill: rgba(18, 27, 40, 0.92);
      stroke: rgba(255,255,255,.16);
      stroke-width: 1.2;
      rx: 8;
    }
    .nav-map-shield text {
      fill: #e4edff;
      font-size: 10px;
      font-weight: 900;
      letter-spacing: .08em;
    }
    .nav-map-hazard,
    .nav-map-impact {
      opacity: 0;
      transition: opacity .2s ease;
    }
    .nav-pane.wrong .nav-map-hazard,
    .nav-pane.impact .nav-map-hazard,
    .nav-pane.impact .nav-map-impact {
      opacity: 1;
    }
    .nav-map-hazard circle,
    .nav-map-impact circle {
      fill: rgba(255, 71, 104, 0.18);
      stroke: rgba(255, 153, 171, 0.85);
      stroke-width: 1.4;
    }
    .nav-map-hazard path { fill: #ffc0cb; }
    .nav-map-impact path { stroke: #ffd1d1; stroke-width: 2.4; fill: none; stroke-linecap: round; }
    .nav-map-car-marker {
      filter: drop-shadow(0 0 14px rgba(64, 139, 255, 0.38));
      transition: transform .16s linear;
    }
    .nav-map-car-marker circle:first-child {
      fill: rgba(50, 136, 255, 0.18);
      stroke: rgba(90, 176, 255, 0.72);
      stroke-width: 1.2;
    }
    .nav-map-car-marker circle:last-child {
      fill: #ffd166;
      stroke: rgba(255,255,255,.35);
      stroke-width: .8;
    }
    .nav-footer {
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: 18px;
      background: rgba(5,8,12,.86);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 16px;
      padding: 12px 14px;
      backdrop-filter: blur(10px);
    }
    .nav-footer .route-name {
      font-size: 18px;
      font-weight: 900;
      margin-bottom: 4px;
    }
    .nav-footer .route-meta {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }
    .scene-pane {
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(circle at 50% 14%, rgba(255,255,255,.06), transparent 14%),
        linear-gradient(180deg, #0a0c11 0%, #05070a 100%);
    }
    .scene-svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    .scene-surface { fill: url(#roadSurface); }
    .scene-shoulder { fill: rgba(255,255,255,.06); }
    .scene-median { fill: url(#medianPaint); }
    .scene-lane-solid { stroke: rgba(255,255,255,.9); stroke-width: 4; fill: none; opacity: .9; }
    .scene-lane-dashed { stroke: rgba(255,255,255,.76); stroke-width: 3.2; stroke-dasharray: 18 22; fill: none; opacity: .88; }
    .scene-route { fill: rgba(42,124,255,.55); stroke: rgba(90,176,255,.55); stroke-width: 2; }
    .scene-bike { fill: rgba(73,255,188,.22); stroke: rgba(73,255,188,.4); stroke-width: 1.4; opacity: 0; transition: opacity .2s ease; }
    .scene-construction { opacity: 0; transition: opacity .2s ease; }
    .construction-block { fill: rgba(255,96,56,.24); stroke: rgba(255,138,61,.62); stroke-width: 2.2; }
    .construction-cone { fill: #ff8a3d; }
    .construction-cone-cap { fill: #fff3dd; }
    .construction-barrier { fill: rgba(255,168,74,.28); stroke: rgba(255,204,115,.56); stroke-width: 1.6; }
    .construction-stripe { stroke: #ffd166; stroke-width: 5; stroke-dasharray: 10 8; }
    .scene-pane.bike .scene-bike { opacity: 1; }
    .scene-pane.construction .scene-construction { opacity: 1; }
    .scene-pane.impact .scene-route {
      fill: rgba(255, 103, 92, 0.32);
      stroke: rgba(255, 170, 122, 0.72);
    }
    .hud-top {
      position: absolute;
      left: 24px;
      right: 24px;
      top: 18px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      z-index: 3;
      pointer-events: none;
    }
    .hud-speed {
      display: flex;
      align-items: baseline;
      gap: 10px;
      background: rgba(5,8,12,.72);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 18px;
      padding: 10px 16px;
      backdrop-filter: blur(10px);
    }
    .hud-speed span {
      font-size: 54px;
      font-weight: 900;
      line-height: 1;
    }
    .hud-speed small {
      font-size: 14px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .hud-meta {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .scene-chip, .speed-sign {
      background: rgba(5,8,12,.72);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 16px;
      padding: 10px 12px;
      backdrop-filter: blur(10px);
    }
    .scene-chip strong {
      display: block;
      font-size: 12px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: #dbe7fb;
      margin-bottom: 3px;
    }
    .scene-chip span {
      display: block;
      font-size: 12px;
      color: var(--muted);
    }
    .speed-sign {
      width: 72px;
      text-align: center;
      background: rgba(245,246,248,.92);
      color: #101317;
      border-color: rgba(0,0,0,.08);
    }
    .speed-sign strong {
      display: block;
      font-size: 11px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .speed-sign span {
      display: block;
      font-size: 30px;
      line-height: 1;
      font-weight: 900;
      margin: 4px 0;
    }
    .warning-banner, .scene-status-chip {
      position: absolute;
      left: 50%;
      transform: translateX(-50%);
      z-index: 3;
      border-radius: 999px;
      padding: 8px 16px;
      letter-spacing: .08em;
      text-transform: uppercase;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,.06);
      pointer-events: none;
    }
    .warning-banner {
      top: 98px;
      background: rgba(255,59,95,.16);
      color: #ff9aaa;
      display: none;
      font-weight: 900;
      box-shadow: 0 0 26px rgba(255,59,95,.2);
    }
    .scene-status-chip {
      top: 140px;
      background: rgba(5,8,12,.72);
      color: #dbe7fb;
      font-size: 11px;
    }
    .impact-banner {
      position: absolute;
      left: 50%;
      top: 176px;
      transform: translateX(-50%);
      z-index: 3;
      padding: 7px 14px;
      border-radius: 999px;
      background: rgba(255,87,87,.2);
      color: #ffd6d6;
      border: 1px solid rgba(255,148,148,.38);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
      font-weight: 800;
      display: none;
      box-shadow: 0 0 20px rgba(255,87,87,.18);
    }
    .car-layer {
      position: absolute;
      inset: 0;
      z-index: 2;
    }
    .collision-cloud {
      position: absolute;
      width: 110px;
      height: 110px;
      border-radius: 50%;
      background:
        radial-gradient(circle, rgba(255,87,87,.42) 0%, rgba(255,152,0,.18) 36%, rgba(255,87,87,0) 72%);
      border: 2px solid rgba(255,120,120,.42);
      transform: translate(-50%, -50%);
      pointer-events: none;
      box-shadow: 0 0 36px rgba(255,87,87,.22);
      display: none;
      z-index: 2;
    }
    .collision-label {
      position: absolute;
      transform: translate(-50%, calc(-100% - 12px));
      background: rgba(30,10,10,.94);
      color: #ffd7d7;
      border: 1px solid rgba(255,120,120,.35);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 10px;
      letter-spacing: .08em;
      text-transform: uppercase;
      white-space: nowrap;
      display: none;
      z-index: 3;
    }
    .scene-pane.impact .collision-cloud {
      box-shadow: 0 0 46px rgba(255,87,87,.38);
      background:
        radial-gradient(circle, rgba(255,87,87,.55) 0%, rgba(255,152,0,.22) 34%, rgba(255,87,87,0) 72%);
    }
    .vehicle {
      position: absolute;
      width: 36px;
      height: 72px;
      transform: translate(-50%, -50%) rotate(var(--rot, 0deg)) scale(var(--scale, 1));
      transition: top .09s linear, left .09s linear, transform .09s linear;
      filter: drop-shadow(0 16px 18px rgba(0,0,0,.5));
      transform-origin: center center;
    }
    .vehicle.suv { width: 40px; height: 78px; }
    .vehicle.truck { width: 44px; height: 92px; }
    .vehicle.bike { width: 18px; height: 42px; }
    .vehicle svg { width: 100%; height: 100%; overflow: visible; }
    .vehicle .car-shadow { fill: rgba(0,0,0,.26); }
    .vehicle .car-body { fill: var(--car-color, #59a6ff); stroke: rgba(255,255,255,.26); stroke-width: 1.4; }
    .vehicle .car-roof { fill: rgba(255,255,255,.16); }
    .vehicle .car-glass { fill: rgba(195,224,255,.78); }
    .vehicle .car-trim { fill: rgba(255,255,255,.12); }
    .vehicle .car-wheel { fill: #090d13; }
    .vehicle .headlight { fill: #fff6c0; opacity: .95; }
    .vehicle .taillight { fill: #ff6c6c; opacity: .92; }
    .vehicle.wrong { --car-color: #ff3b5f; }
    .vehicle.normal { --car-color: #aeb7c6; }
    .vehicle.suv { --car-color: #c3cad7; }
    .vehicle.truck { --car-color: #c1c8d0; }
    .vehicle.bike { --car-color: #c9ffe0; }
    .vehicle.alert-source {
      filter: drop-shadow(0 0 16px rgba(90,176,255,.62)) drop-shadow(0 0 34px rgba(255,59,95,.25)) drop-shadow(0 12px 14px rgba(0,0,0,.42));
    }
    .ripple {
      position: absolute;
      border: 2px solid rgba(66,124,255,.72);
      border-radius: 50%;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      animation: ripple 1.8s infinite;
      display: none;
      z-index: 1;
    }
    .ripple.r1 { width: 130px; height: 130px; }
    .ripple.r2 { width: 220px; height: 220px; animation-delay: .22s; }
    .ripple.r3 { width: 320px; height: 320px; animation-delay: .44s; }
    @keyframes ripple { from { opacity: .85; } to { opacity: .12; } }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid rgba(35,50,71,.75); padding: 9px 6px; text-align: left; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .pill { padding: 4px 8px; border-radius: 999px; font-size: 11px; font-weight: 800; }
    .active { background: rgba(41,241,156,.13); color: var(--green); }
    .idle { background: rgba(127,141,163,.13); color: #a8b3c5; }
    .suppressed { background: rgba(255,209,102,.15); color: var(--yellow); }
    .failed { background: rgba(255,59,95,.16); color: var(--red); }
    .timeline { max-height: 330px; overflow: auto; padding-right: 4px; }
    .alert {
      border-left: 2px solid var(--red);
      padding: 0 0 12px 12px;
      margin: 0 0 8px 6px;
    }
    .alert .t { color: #ff9aaa; font-size: 12px; font-weight: 800; }
    .alert .m { font-size: 13px; color: var(--muted); }
    .small { color: var(--muted); font-size: 12px; line-height: 1.45; }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .map { grid-template-columns: 1fr; height: 760px; }
      .nav-pane { min-height: 210px; }
      .nav-route { height: 240px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SwarmMind Safety Net</h1>
      <div class="sub">Toolbox3-inspired diagnostic simulation · 9-layer wrong-way detection · V2X swarm safety</div>
    </div>
    <div class="sub">API: <a style="color:#59a6ff" href="/docs">/docs</a> · <a style="color:#59a6ff" href="/state">/state</a></div>
  </header>

  <main class="grid">
    <section class="panel">
      <h2>CSV Trace Player</h2>
      <div id="status" class="status NORMAL">NORMAL</div>
      <label>Trace</label>
      <select id="csvSelect"></select>
      <label>Delay per frame (ms)</label>
      <input id="delay" type="number" min="0" max="1000" value="35" />
      <label>Demo length (seconds)</label>
      <input id="maxSeconds" type="number" min="1" max="3600" value="90" />
      <button id="playBtn">PLAY TRACE</button>
      <button id="refreshBtn" class="secondary">RESET SCENE</button>
      <label style="margin-top:12px"><input id="collisionToggle" type="checkbox" checked style="width:auto; margin-right:8px">Show collision cloud</label>
      <label><input id="impactToggle" type="checkbox" checked style="width:auto; margin-right:8px">Show impact response</label>

      <h2 style="margin-top:22px">Config Editor</h2>
      <label>LNN threshold</label><input id="lnn" type="number" min="0" max="1" step="0.01" />
      <label>ODE threshold</label><input id="ode" type="number" min="0" max="1" step="0.01" />
      <label>TTI threshold</label><input id="tti" type="number" min="0" max="1" step="0.01" />
      <label>Wrong-way heading delta</label><input id="delta" type="number" min="0" max="180" step="1" />
      <label>Min wrong-way steps</label><input id="steps" type="number" min="1" max="500" step="1" />
      <button id="saveBtn" class="secondary">SAVE CONFIG</button>
      <p class="small">Config is in-memory for deterministic simulation sessions. Same CSV + same config produces the same replay.</p>
    </section>

    <section>
      <div class="metrics">
        <div class="metric"><div class="k">Time</div><div id="metricTime" class="v">0.0s</div></div>
        <div class="metric"><div class="k">Vehicles</div><div id="metricVehicles" class="v">0</div></div>
        <div class="metric"><div class="k">Alerts</div><div id="metricAlerts" class="v">0</div></div>
        <div class="metric"><div class="k">Active CSV</div><div id="metricCsv" class="v" style="font-size:17px">none</div></div>
      </div>
      <div class="panel">
        <h2>Central Simulation Map</h2>
        <div class="map">
          <div id="navPane" class="nav-pane">
            <div class="nav-label">Route Overview</div>
            <div class="nav-route">
              <div id="mapboxRouteMap"></div>
              <div id="mapboxStatus" class="mapbox-status">Vector preview</div>
              <div class="nav-grid"></div>
              <svg viewBox="0 0 260 420" preserveAspectRatio="none">
                <path class="nav-map-minor" d="M8 328 C42 304 80 286 124 254 C164 224 194 186 248 142"></path>
                <path class="nav-map-minor" d="M16 234 C58 220 102 198 134 174 C162 152 190 124 232 82"></path>
                <path class="nav-map-minor" d="M56 408 C90 362 110 330 140 304 C176 272 214 250 252 220"></path>
                <path class="nav-map-minor" d="M154 408 C166 360 178 308 188 242 C198 180 214 124 242 52"></path>
                <path class="nav-map-road" d="M30 394 C62 350 76 320 90 280 C102 244 122 208 144 176 C168 142 186 104 214 38"></path>
                <path class="nav-map-road" d="M188 404 C196 352 198 300 204 240 C210 174 220 114 244 50"></path>
                <g class="nav-map-shield" transform="translate(148 92)">
                  <rect x="0" y="0" width="42" height="22"></rect>
                  <text x="21" y="14" text-anchor="middle">I-405</text>
                </g>
                <text class="nav-map-label" x="114" y="150">Downtown Corridor</text>
                <text class="nav-map-label" x="38" y="300">Service Road</text>
                <path id="routePathShadow" class="nav-map-route-shadow" d=""></path>
                <path id="routePath" class="nav-map-route" d=""></path>
                <path id="detourPath" class="nav-map-detour" d=""></path>
                <path id="bikePath" class="nav-map-bike" d=""></path>
                <g id="navHazard" class="nav-map-hazard" transform="translate(132 184)">
                  <circle cx="0" cy="0" r="12"></circle>
                  <path d="M-3 -6 L3 -6 L2 1 L-2 1 Z M0 4.5 A1.4 1.4 0 1 1 -0.01 4.5"></path>
                </g>
                <g id="navImpact" class="nav-map-impact" transform="translate(150 218)">
                  <circle cx="0" cy="0" r="16"></circle>
                  <path d="M-7 -7 L7 7 M7 -7 L-7 7"></path>
                </g>
                <g id="navCarMarker" class="nav-map-car-marker" transform="translate(30 394)">
                  <circle cx="0" cy="0" r="12"></circle>
                  <circle cx="0" cy="0" r="5"></circle>
                </g>
                <circle cx="30" cy="394" r="8" fill="#2f8fff"></circle>
                <circle cx="214" cy="38" r="7" fill="#ffd166"></circle>
              </svg>
            </div>
            <div class="nav-footer">
              <div id="traceLabel" class="route-name">wrong_way.csv</div>
              <div class="route-meta">
                <span id="sceneHint">4-lane divided highway</span>
                <span id="riskValue">Risk 0%</span>
              </div>
            </div>
          </div>
          <div id="scenePane" class="scene-pane">
            <div class="hud-top">
              <div class="hud-speed"><span id="speedValue">55</span><small>mph</small></div>
              <div class="hud-meta">
                <div class="scene-chip">
                  <strong id="sceneMode">Live Drive View</strong>
                  <span id="sceneStatusText">Normal flow</span>
                </div>
                <div class="speed-sign">
                  <strong>Speed Limit</strong>
                  <span>45</span>
                  <small>MAX</small>
                </div>
              </div>
            </div>
            <div id="warningBanner" class="warning-banner">Wrong-Way Detected</div>
            <div id="sceneStatusChip" class="scene-status-chip">Monitoring Highway Corridor</div>
            <div id="sceneImpactBanner" class="impact-banner">Impact response active · braking + reroute</div>
            <svg class="scene-svg" viewBox="0 0 1000 620" preserveAspectRatio="none">
              <defs>
                <linearGradient id="roadSurface" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#40454d"/>
                  <stop offset="55%" stop-color="#252a31"/>
                  <stop offset="100%" stop-color="#10151b"/>
                </linearGradient>
                <linearGradient id="medianPaint" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#ffd166"/>
                  <stop offset="100%" stop-color="#b8891a"/>
                </linearGradient>
              </defs>
              <rect x="0" y="0" width="1000" height="620" fill="transparent"></rect>
              <polygon class="scene-shoulder" points="282,620 320,580 440,140 418,140"></polygon>
              <polygon class="scene-shoulder" points="560,140 680,580 718,620 582,140"></polygon>
              <polygon class="scene-surface" points="320,580 440,140 560,140 680,580"></polygon>
              <polygon class="scene-median" points="445,580 495,140 505,140 555,580"></polygon>
              <path class="scene-lane-solid" d="M320 580 L440 140"></path>
              <path class="scene-lane-solid" d="M445 580 L495 140"></path>
              <path class="scene-lane-solid" d="M555 580 L505 140"></path>
              <path class="scene-lane-solid" d="M680 580 L560 140"></path>
              <path class="scene-lane-dashed" d="M390 580 L468 140"></path>
              <path class="scene-lane-dashed" d="M610 580 L532 140"></path>
              <polygon id="routeCorridor" class="scene-route" points="396,580 438,580 496,140 474,140"></polygon>
              <polygon id="bikeLaneOverlay" class="scene-bike" points="286,580 316,580 434,140 422,140"></polygon>
              <g id="constructionOverlay" class="scene-construction">
                <polygon class="construction-block" points="386,580 438,580 492,330 456,330"></polygon>
                <polygon class="construction-barrier" points="446,352 474,352 486,318 458,318"></polygon>
                <line class="construction-stripe" x1="402" y1="556" x2="476" y2="302"></line>
                <g transform="translate(438 466)">
                  <circle class="construction-cone" cx="0" cy="0" r="11"></circle>
                  <rect class="construction-cone-cap" x="-5" y="-6" width="10" height="3" rx="1.5"></rect>
                </g>
                <g transform="translate(452 414)">
                  <circle class="construction-cone" cx="0" cy="0" r="10"></circle>
                  <rect class="construction-cone-cap" x="-4" y="-5" width="8" height="3" rx="1.5"></rect>
                </g>
                <g transform="translate(466 364)">
                  <circle class="construction-cone" cx="0" cy="0" r="9"></circle>
                  <rect class="construction-cone-cap" x="-4" y="-5" width="8" height="3" rx="1.5"></rect>
                </g>
              </g>
            </svg>
            <div id="r1" class="ripple r1"></div>
            <div id="r2" class="ripple r2"></div>
            <div id="r3" class="ripple r3"></div>
            <div id="collisionCloud" class="collision-cloud"></div>
            <div id="collisionLabel" class="collision-label"></div>
            <div id="carLayer" class="car-layer"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>9-Layer Inspector</h2>
      <table>
        <thead><tr><th>Layer</th><th>Confidence</th><th>Status</th></tr></thead>
        <tbody id="layersBody"></tbody>
      </table>
      <h2 style="margin-top:22px">Alert Log Timeline</h2>
      <div id="timeline" class="timeline"><p class="small">No alerts yet. Play a wrong-way trace.</p></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const MAPBOX_TOKEN = __MAPBOX_TOKEN_JSON__;
    const MAPBOX_STYLE = __MAPBOX_STYLE_JSON__;
    const MAPBOX_ENABLED = Boolean(MAPBOX_TOKEN) && typeof mapboxgl !== "undefined";
    const NAV_ROUTE_POINTS = {
      default: [[30, 394], [58, 350], [80, 304], [104, 252], [134, 204], [162, 154], [188, 104], [214, 38]],
      construction: [[30, 394], [58, 350], [80, 304], [110, 268], [150, 246], [188, 210], [212, 160], [214, 38]],
      bike: [[40, 398], [62, 362], [78, 322], [94, 274], [114, 222], [140, 176], [170, 122], [202, 52]],
    };
    const NAV_DETOUR_POINTS = {
      construction: [[80, 304], [114, 286], [148, 274], [182, 252], [206, 214], [222, 170]],
      impact: [[132, 204], [156, 190], [184, 182], [214, 170], [232, 140]],
    };
    let mapboxMap = null;
    let mapboxReady = false;
    let pendingMapState = null;
    let alerts = [];
    function lerp(a, b, t) {
      return a + ((b - a) * t);
    }

    function pathFromPoints(points) {
      if (!points?.length) return "";
      return `M ${points.map(([x, y]) => `${x} ${y}`).join(" L ")}`;
    }

    function pointAlongPoints(points, ratio) {
      if (!points?.length) return { x: 30, y: 394 };
      if (points.length === 1) return { x: points[0][0], y: points[0][1] };
      const clamped = Math.max(0, Math.min(1, ratio));
      const segments = [];
      let total = 0;
      for (let i = 0; i < points.length - 1; i += 1) {
        const [x1, y1] = points[i];
        const [x2, y2] = points[i + 1];
        const len = Math.hypot(x2 - x1, y2 - y1);
        segments.push({ x1, y1, x2, y2, len });
        total += len;
      }
      let target = total * clamped;
      for (const seg of segments) {
        if (target <= seg.len) {
          const t = seg.len === 0 ? 0 : target / seg.len;
          return { x: lerp(seg.x1, seg.x2, t), y: lerp(seg.y1, seg.y2, t) };
        }
        target -= seg.len;
      }
      const last = points[points.length - 1];
      return { x: last[0], y: last[1] };
    }

    function routeProgress(vehicle) {
      if (!vehicle) return 0.08;
      const progress = Math.max(0, Math.min(100, Number(vehicle.progress ?? 0)));
      return vehicle.direction === "southbound" ? 1 - (progress / 100) : progress / 100;
    }

    function roadFrame(depth) {
      const clamped = Math.max(8, Math.min(96, depth));
      const t = clamped / 100;
      return {
        t,
        y: lerp(564, 150, t),
        leftOuter: lerp(320, 440, t),
        leftDivider: lerp(390, 468, t),
        medianLeft: lerp(445, 495, t),
        medianRight: lerp(555, 505, t),
        rightDivider: lerp(610, 532, t),
        rightOuter: lerp(680, 560, t),
      };
    }

    function laneEdgesAtDepth(lane, depth) {
      const frame = roadFrame(depth);
      const normalizedLane = Number.isFinite(Number(lane)) ? Number(lane) : 1;
      if (normalizedLane < 2) {
        if (normalizedLane === 0) return { left: frame.leftOuter + 6, right: frame.leftDivider - 4, y: frame.y, frame };
        return { left: frame.leftDivider + 4, right: frame.medianLeft - 8, y: frame.y, frame };
      }
      if (normalizedLane === 2) return { left: frame.medianRight + 8, right: frame.rightDivider - 4, y: frame.y, frame };
      return { left: frame.rightDivider + 4, right: frame.rightOuter - 6, y: frame.y, frame };
    }

    function scenePercent(point) {
      return {
        x: (point.x / 1000) * 100,
        y: (point.y / 620) * 100,
      };
    }

    function perspectivePosition(vehicle) {
      const progress = Number(vehicle.progress ?? 50);
      const depth = vehicle.direction === "northbound" ? 100 - progress : progress;
      const clamped = Math.max(8, Math.min(96, depth));
      const edges = laneEdgesAtDepth(vehicle.lane ?? 1, clamped);
      let x = (edges.left + edges.right) / 2;
      if (vehicle.behavior?.includes("bike")) {
        x = edges.frame.leftOuter - 20;
      }
      const point = scenePercent({ x, y: edges.y });
      const scale = 0.34 + edges.frame.t * 0.56;
      return { x: point.x, y: point.y, scale, depth: clamped };
    }

    function initializeMapbox() {
      if (!MAPBOX_ENABLED || mapboxMap) {
        if (!MAPBOX_ENABLED) {
          $("mapboxStatus").textContent = "Vector preview";
        }
        return;
      }
      mapboxgl.accessToken = MAPBOX_TOKEN;
      mapboxMap = new mapboxgl.Map({
        container: "mapboxRouteMap",
        style: MAPBOX_STYLE,
        attributionControl: true,
        pitchWithRotate: false,
        dragRotate: false,
        touchPitch: false,
        cooperativeGestures: true,
      });
      mapboxMap.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "bottom-right");
      mapboxMap.on("load", () => {
        mapboxReady = true;
        $("mapboxStatus").textContent = "Live map";
        $("mapboxStatus").classList.add("live");
        $("navPane").querySelector(".nav-route").classList.add("has-map");
        mapboxMap.addSource("swarmmind-route", {
          type: "geojson",
          data: { type: "Feature", geometry: { type: "LineString", coordinates: [] } },
        });
        mapboxMap.addSource("swarmmind-vehicle", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        mapboxMap.addSource("swarmmind-hazard", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        mapboxMap.addLayer({
          id: "swarmmind-route-glow",
          type: "line",
          source: "swarmmind-route",
          paint: {
            "line-color": "#5aaeff",
            "line-width": 10,
            "line-opacity": 0.22,
          },
        });
        mapboxMap.addLayer({
          id: "swarmmind-route-line",
          type: "line",
          source: "swarmmind-route",
          paint: {
            "line-color": "#2f8fff",
            "line-width": 5,
          },
        });
        mapboxMap.addLayer({
          id: "swarmmind-hazard-ring",
          type: "circle",
          source: "swarmmind-hazard",
          paint: {
            "circle-radius": 16,
            "circle-color": "#ff4f6a",
            "circle-opacity": 0.16,
            "circle-stroke-width": 2,
            "circle-stroke-color": "#ff9aaa",
          },
        });
        mapboxMap.addLayer({
          id: "swarmmind-hazard-core",
          type: "circle",
          source: "swarmmind-hazard",
          paint: {
            "circle-radius": 6,
            "circle-color": "#ffd7d7",
            "circle-stroke-width": 2,
            "circle-stroke-color": "#ff4f6a",
          },
        });
        mapboxMap.addLayer({
          id: "swarmmind-vehicle",
          type: "circle",
          source: "swarmmind-vehicle",
          paint: {
            "circle-radius": 6,
            "circle-color": "#ffd166",
            "circle-stroke-width": 2,
            "circle-stroke-color": "#2f8fff",
          },
        });
        if (pendingMapState) {
          updateMapboxRoute(
            pendingMapState.state,
            pendingMapState.focusVehicle,
            pendingMapState.collision,
            pendingMapState.isWrongWay,
            pendingMapState.isConstruction,
            pendingMapState.showImpact,
          );
        }
      });
      mapboxMap.on("error", () => {
        $("mapboxStatus").textContent = "Map load issue";
      });
    }

    function setGeoJson(sourceId, data) {
      if (!mapboxReady || !mapboxMap?.getSource(sourceId)) return;
      mapboxMap.getSource(sourceId).setData(data);
    }

    function updateMapboxRoute(state, focusVehicle, collision, isWrongWay, isConstruction, showImpact) {
      if (!MAPBOX_ENABLED) return;
      pendingMapState = { state, focusVehicle, collision, isWrongWay, isConstruction, showImpact };
      initializeMapbox();
      if (!mapboxReady) return;
      const routeGeometry = Array.isArray(state.route_geometry) ? state.route_geometry : [];
      const center = Array.isArray(state.map_center) ? state.map_center : (focusVehicle ? [focusVehicle.lon, focusVehicle.lat] : null);
      const bounds = Array.isArray(state.map_bounds) ? state.map_bounds : null;
      setGeoJson("swarmmind-route", {
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: routeGeometry,
        },
      });
      const focusPoint = focusVehicle ? [Number(focusVehicle.lon), Number(focusVehicle.lat)] : null;
      setGeoJson("swarmmind-vehicle", {
        type: "FeatureCollection",
        features: focusPoint ? [{
          type: "Feature",
          geometry: { type: "Point", coordinates: focusPoint },
        }] : [],
      });
      const hazardPoint = focusPoint;
      setGeoJson("swarmmind-hazard", {
        type: "FeatureCollection",
        features: (collision?.active && hazardPoint) ? [{
          type: "Feature",
          geometry: { type: "Point", coordinates: hazardPoint },
        }] : [],
      });
      const routeColor = showImpact ? "#ff7a6c" : isConstruction ? "#ffbf5a" : isWrongWay ? "#2f8fff" : "#5aaeff";
      if (mapboxMap.getLayer("swarmmind-route-line")) {
        mapboxMap.setPaintProperty("swarmmind-route-line", "line-color", routeColor);
      }
      if (mapboxMap.getLayer("swarmmind-route-glow")) {
        mapboxMap.setPaintProperty("swarmmind-route-glow", "line-color", routeColor);
      }
      if (bounds && bounds.length === 2) {
        mapboxMap.fitBounds(bounds, { padding: 36, duration: 600, maxZoom: 15 });
      } else if (center) {
        mapboxMap.easeTo({ center, zoom: 14, duration: 600 });
      }
    }

    function corridorPointsForLane(lane, mode = "default") {
      if (mode === "bike") {
        const near = roadFrame(96);
        const far = roadFrame(10);
        return `${near.leftOuter - 30},${near.y} ${near.leftOuter - 2},${near.y} ${far.leftOuter - 6},${far.y} ${far.leftOuter - 22},${far.y}`;
      }
      if (mode === "construction") {
        const near = laneEdgesAtDepth(0, 96);
        const midNear = laneEdgesAtDepth(0, 70);
        const midFar = laneEdgesAtDepth(1, 42);
        const far = laneEdgesAtDepth(1, 12);
        return `${near.left},${near.y} ${near.right},${near.y} ${midNear.right},${midNear.y} ${midFar.right},${midFar.y} ${far.right},${far.y} ${far.left},${far.y} ${midFar.left},${midFar.y} ${midNear.left},${midNear.y}`;
      }
      const near = laneEdgesAtDepth(lane, 96);
      const far = laneEdgesAtDepth(lane, 10);
      return `${near.left},${near.y} ${near.right},${near.y} ${far.right},${far.y} ${far.left},${far.y}`;
    }

    function vehicleSvg(className) {
      if (className === "bike") {
        return `
          <svg viewBox="0 0 28 60" class="${className}">
            <ellipse class="car-shadow" cx="14" cy="50" rx="12" ry="6"></ellipse>
            <circle class="car-wheel" cx="8" cy="44" r="5"></circle>
            <circle class="car-wheel" cx="20" cy="44" r="5"></circle>
            <path class="car-trim" d="M8 44 L13 28 L19 28 L20 44" stroke="rgba(255,255,255,.35)" stroke-width="2" fill="none"></path>
            <circle class="car-body" cx="14" cy="18" r="6"></circle>
            <path class="car-trim" d="M14 24 L14 34" stroke="rgba(255,255,255,.35)" stroke-width="2"></path>
          </svg>
        `;
      }
      return `
        <svg viewBox="0 0 52 104" class="${className}">
          <ellipse class="car-shadow" cx="26" cy="86" rx="18" ry="12"></ellipse>
          <rect class="car-wheel" x="8" y="20" rx="3" width="6" height="18"></rect>
          <rect class="car-wheel" x="38" y="20" rx="3" width="6" height="18"></rect>
          <rect class="car-wheel" x="8" y="66" rx="3" width="6" height="18"></rect>
          <rect class="car-wheel" x="38" y="66" rx="3" width="6" height="18"></rect>
          <path class="car-body" d="M17 6 C19 2, 33 2, 35 6 L42 24 L42 77 C42 89, 34 97, 26 98 C18 97, 10 89, 10 77 L10 24 Z"></path>
          <path class="car-roof" d="M19 18 C20 12, 32 12, 33 18 L36 38 L16 38 Z"></path>
          <rect class="car-glass" x="16" y="41" rx="6" ry="6" width="20" height="22"></rect>
          <rect class="car-trim" x="18" y="67" rx="5" ry="5" width="16" height="12"></rect>
          <rect class="headlight" x="16" y="8" rx="2" width="7" height="4"></rect>
          <rect class="headlight" x="29" y="8" rx="2" width="7" height="4"></rect>
          <rect class="taillight" x="16" y="90" rx="2" width="7" height="4"></rect>
          <rect class="taillight" x="29" y="90" rx="2" width="7" height="4"></rect>
        </svg>
      `;
    }

    function statusClass(status) {
      if (status.includes("WRONG")) return "status WRONG";
      if (status.includes("CAUTION")) return "status CAUTION";
      return "status NORMAL";
    }

    function layerPill(status) {
      return `<span class="pill ${status}">${status}</span>`;
    }

    function updateState(state, eventPayload=null) {
      const status = eventPayload?.status || state.status || "NORMAL";
      const csvName = eventPayload?.session_meta?.csv_name || state.active_csv || "none";
      const vehicles = eventPayload?.vehicles || state.vehicles || [];
      const focusVehicle = vehicles.find(v => v.is_alert_source) || vehicles[0];
      const collision = eventPayload?.collision || state.collision;
      const isBike = csvName.includes("bike") || vehicles.some(v => v.behavior?.includes("bike"));
      const isConstruction = csvName.includes("construction") || csvName.includes("detour");
      const isWrongWay = status.includes("WRONG");
      const isCaution = status.includes("CAUTION");
      const showImpact = $("impactToggle").checked && collision?.active;
      const navRoutePoints = isBike ? NAV_ROUTE_POINTS.bike : isConstruction ? NAV_ROUTE_POINTS.construction : NAV_ROUTE_POINTS.default;
      const detourPoints = isConstruction ? NAV_DETOUR_POINTS.construction : showImpact ? NAV_DETOUR_POINTS.impact : [];
      const navProgress = routeProgress(focusVehicle);
      const navMarker = pointAlongPoints(navRoutePoints, navProgress);
      const navHazardPoint = pointAlongPoints(navRoutePoints, showImpact ? Math.min(0.78, navProgress + 0.06) : Math.max(0.42, navProgress));

      $("status").className = statusClass(status);
      $("status").textContent = status;
      $("metricTime").textContent = `${Number(eventPayload?.time ?? state.time ?? 0).toFixed(1)}s`;
      $("metricVehicles").textContent = vehicles.length;
      $("metricAlerts").textContent = (state.alerts || alerts).length;
      $("metricCsv").textContent = csvName;
      $("traceLabel").textContent = csvName;
      $("speedValue").textContent = Math.round(Number(focusVehicle?.speed || 0));
      $("riskValue").textContent = `Risk ${(Number(collision?.risk_score || 0) * 100).toFixed(0)}%`;
      $("sceneMode").textContent = showImpact ? "Impact Response" : isWrongWay ? "Wrong-Way Watch" : isConstruction ? "Detour Simulation" : isBike ? "Bike Path Filter" : "Live Drive View";
      $("sceneStatusText").textContent = showImpact ? "Auto-brake, V2X broadcast, reroute guidance" : isWrongWay ? "Opposing car on same carriageway" : isConstruction ? "Lane closure and reroute visible" : isBike ? "Micromobility suppression active" : isCaution ? "Anomaly under review" : "Normal divided-highway flow";
      $("sceneHint").textContent = showImpact ? "Collision hot-zone rerouted on map" : isWrongWay ? "Collision corridor highlighted" : isConstruction ? "Temporary taper and lane shift" : isBike ? "Protected side path isolated from highway" : "4-lane divided highway";
      $("sceneStatusChip").textContent = showImpact ? "Emergency brake · reroute active" : isWrongWay ? "Safety corridor active" : isConstruction ? "Construction taper active" : isBike ? "Bike lane guard active" : isCaution ? "Tracking anomaly" : "Monitoring highway corridor";
      $("warningBanner").textContent = showImpact ? "Collision Response Active" : isWrongWay ? "Wrong-Way Detected" : isConstruction ? "Construction Detour Active" : "Monitoring Corridor";
      $("warningBanner").style.display = (showImpact || isWrongWay || isConstruction) ? "block" : "none";
      $("sceneImpactBanner").style.display = showImpact ? "block" : "none";
      $("scenePane").classList.toggle("bike", isBike);
      $("scenePane").classList.toggle("construction", isConstruction);
      $("scenePane").classList.toggle("impact", showImpact);
      $("navPane").classList.toggle("bike", isBike);
      $("navPane").classList.toggle("construction", isConstruction);
      $("navPane").classList.toggle("wrong", isWrongWay);
      $("navPane").classList.toggle("impact", showImpact);
      updateMapboxRoute(state, focusVehicle, collision, isWrongWay, isConstruction, showImpact);
      $("routeCorridor").setAttribute("points", corridorPointsForLane(focusVehicle?.lane ?? 1, isBike ? "bike" : isConstruction ? "construction" : "default"));
      $("routePathShadow").setAttribute("d", pathFromPoints(navRoutePoints));
      $("routePath").setAttribute("d", pathFromPoints(navRoutePoints));
      $("detourPath").setAttribute("d", pathFromPoints(detourPoints));
      $("detourPath").style.opacity = (isConstruction || showImpact) ? "1" : "0";
      $("detourPath").style.stroke = showImpact ? "#ff8f8f" : "#ffbf5a";
      $("bikePath").setAttribute("d", pathFromPoints(NAV_ROUTE_POINTS.bike));
      $("bikePath").style.opacity = isBike ? "1" : "0";
      $("navCarMarker").setAttribute("transform", `translate(${navMarker.x} ${navMarker.y})`);
      $("navHazard").setAttribute("transform", `translate(${navHazardPoint.x} ${navHazardPoint.y})`);
      $("navImpact").setAttribute("transform", `translate(${navHazardPoint.x + 20} ${navHazardPoint.y + 18})`);

      $("carLayer").innerHTML = vehicles.map(vehicle => {
        const pos = perspectivePosition(vehicle);
        const directionRot = vehicle.direction === "southbound" ? 180 : 0;
        const roleClass = vehicle.is_alert_source ? "alert-source" : "";
        const spriteClass = vehicle.sprite || "normal";
        const behavior = vehicle.behavior || "";
        const behaviorClass = behavior.includes("wrong") ? "wrong" : behavior.includes("bike") ? "bike" : spriteClass;
        return `
          <div
            class="vehicle ${behaviorClass} ${roleClass}"
            style="left:${pos.x}%; top:${pos.y}%; --rot:${directionRot}deg; --scale:${pos.scale}"
            title="${vehicle.id} · ${vehicle.behavior} · ${vehicle.speed} km/h"
          >${vehicleSvg(spriteClass)}</div>
        `;
      }).join("");

      const hasAlert = Boolean(eventPayload?.alert) || status.includes("WRONG");
      const focusPos = perspectivePosition(focusVehicle || { lane: 1, progress: 50, direction: "northbound" });
      const rippleLeft = `${focusPos.x}%`;
      const rippleTop = `${focusPos.y}%`;
      ["r1","r2","r3"].forEach(id => {
        $(id).style.display = hasAlert ? "block" : "none";
        $(id).style.left = rippleLeft;
        $(id).style.top = rippleTop;
      });

      const showCollision = $("collisionToggle").checked && collision?.active;
      const collisionPos = perspectivePosition({
        lane: collision?.lane ?? focusVehicle?.lane ?? 1,
        progress: collision?.progress ?? focusVehicle?.progress ?? 50,
        direction: focusVehicle?.direction ?? "northbound"
      });
      const collisionLeft = `${collisionPos.x}%`;
      const collisionTop = `${collisionPos.y}%`;
      $("collisionCloud").style.display = showCollision ? "block" : "none";
      $("collisionCloud").style.left = collisionLeft;
      $("collisionCloud").style.top = collisionTop;
      $("collisionLabel").style.display = showCollision ? "block" : "none";
      $("collisionLabel").style.left = collisionLeft;
      $("collisionLabel").style.top = collisionTop;
      $("collisionLabel").textContent = showImpact
        ? `Impact response · ${collision?.impact_count || 0} vehicles rerouted`
        : `Collision Risk ${(Number(collision?.risk_score || 0) * 100).toFixed(0)}%`;

      const layers = eventPayload?.layers || state.layers || [];
      $("layersBody").innerHTML = layers.map(l => `
        <tr>
          <td>${l.layer}</td>
          <td>${Number(l.confidence).toFixed(3)}</td>
          <td>${layerPill(l.status)}</td>
        </tr>
      `).join("");

      if (eventPayload?.alert) alerts.push(eventPayload.alert);
      if (state.alerts) alerts = state.alerts;
      renderTimeline();
    }

    function renderTimeline() {
      if (!alerts.length) {
        $("timeline").innerHTML = `<p class="small">No alerts yet. Play a wrong-way trace.</p>`;
        return;
      }
      $("timeline").innerHTML = [...alerts].reverse().slice(0, 30).map(a => `
        <div class="alert">
          <div class="t">t=${a.time}s · confidence=${a.confidence}</div>
          <strong>${a.type}</strong>
          <div class="m">${a.location} · V2X radius ${a.v2x_radius_km} km</div>
        </div>
      `).join("");
    }

    async function loadInitial() {
      const csvResp = await fetch("/csvs");
      const csvs = (await csvResp.json()).csvs;
      $("csvSelect").innerHTML = csvs.map(c => `<option value="${c}">${c}</option>`).join("");
      if (csvs.includes("wrong_way.csv")) $("csvSelect").value = "wrong_way.csv";

      const config = await (await fetch("/config")).json();
      $("lnn").value = config.lnn_threshold;
      $("ode").value = config.ode_threshold;
      $("tti").value = config.tti_threshold;
      $("delta").value = config.wrong_way_delta;
      $("steps").value = config.min_wrong_steps;

      await resetScene();
    }

    async function resetScene() {
      alerts = [];
      const response = await fetch("/trace/reset", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          csv_filename: $("csvSelect").value,
          max_seconds: Number($("maxSeconds").value)
        })
      });
      const state = await response.json();
      updateState(state);
    }

    async function refreshState() {
      const state = await (await fetch("/state")).json();
      updateState(state);
    }

    async function saveConfig() {
      await fetch("/config", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          lnn_threshold: Number($("lnn").value),
          ode_threshold: Number($("ode").value),
          tti_threshold: Number($("tti").value),
          wrong_way_delta: Number($("delta").value),
          min_wrong_steps: Number($("steps").value)
        })
      });
      await refreshState();
    }

    async function playTrace() {
      $("playBtn").disabled = true;
      alerts = [];
      const response = await fetch("/playback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          csv_filename: $("csvSelect").value,
          delay_ms: Number($("delay").value),
          max_seconds: Number($("maxSeconds").value)
        })
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\\n\\n");
        buffer = chunks.pop();
        for (const chunk of chunks) {
          const line = chunk.split("\\n").find(l => l.startsWith("data: "));
          if (!line) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.event === "playback-complete") continue;
      const state = await (await fetch("/state")).json();
      updateState(state, payload);
        }
      }
      $("playBtn").disabled = false;
    }

    $("playBtn").addEventListener("click", playTrace);
    $("refreshBtn").addEventListener("click", resetScene);
    $("saveBtn").addEventListener("click", saveConfig);
    $("csvSelect").addEventListener("change", resetScene);
    $("collisionToggle").addEventListener("change", refreshState);
    $("impactToggle").addEventListener("change", refreshState);
    loadInitial();
  </script>
</body>
</html>
"""

ROOT_HTML = (
    ROOT_HTML
    .replace("__MAPBOX_TOKEN_JSON__", json.dumps(MAPBOX_PUBLIC_TOKEN))
    .replace("__MAPBOX_STYLE_JSON__", json.dumps(MAPBOX_STYLE))
)


@app.get("/", response_class=HTMLResponse)
def root_panel() -> str:
    return ROOT_HTML


@app.get("/health")
def health() -> dict[str, Any]:
    readiness = readiness_report(SETTINGS, ROOT)
    return {
        "status": "ok" if readiness["database"]["ready"] else "degraded",
        "environment": SETTINGS.environment,
        "auth_mode": SETTINGS.auth_mode,
        "database_target": SETTINGS.database_target,
    }


@app.get("/readiness")
def get_readiness() -> dict[str, Any]:
    return readiness_report(SETTINGS, ROOT)


@app.get("/dependencies")
def get_dependencies() -> dict[str, Any]:
    report = readiness_report(SETTINGS, ROOT)
    return {
        "database": report["database"],
        "auth": report["auth"],
        "feeds": report["feeds"],
        "blockers": report["blockers"],
    }


@app.get("/state")
def get_state(session_id: str | None = None) -> dict[str, Any]:
    try:
        return SERVICE.get_state(session_id)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/config")
def get_config() -> dict[str, Any]:
    return SERVICE.get_config()


@app.post("/config")
def update_config(update: BackendConfigUpdate) -> dict[str, Any]:
    try:
        saved = SERVICE.update_config(update.model_dump(exclude_none=True))
        response = dict(saved["config"])
        response["_version"] = saved["version"]
        return response
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/config/history")
def get_config_history(limit: int = 100) -> dict[str, Any]:
    return {"versions": SERVICE.get_config_history(limit=limit)}


@app.post("/config/rollback")
def rollback_config(request: ConfigRollbackRequest) -> dict[str, Any]:
    try:
        saved = SERVICE.rollback_config(request.version)
        response = dict(saved["config"])
        response["_version"] = saved["version"]
        response["_rolled_back_from"] = request.version
        return response
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/presets")
def list_presets() -> dict[str, Any]:
    return {"presets": SERVICE.list_presets()}


@app.post("/presets")
def save_preset(request: PresetRequest) -> dict[str, Any]:
    try:
        return SERVICE.save_preset(request.preset_name)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/csvs")
def list_csvs() -> dict[str, list[str]]:
    return {"csvs": SERVICE.list_csvs()}


@app.get("/sessions")
def list_sessions() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": SERVICE.list_sessions()}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    return SERVICE.get_metrics()


@app.get("/state/{session_id}")
def get_session_state(session_id: str) -> dict[str, Any]:
    try:
        return SERVICE.get_state(session_id)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/alerts/{session_id}")
def get_session_alerts(session_id: str) -> dict[str, Any]:
    try:
        return {"session_id": session_id, "alerts": SERVICE.get_alerts(session_id)}
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/events/{session_id}")
def get_session_events(session_id: str, start_frame: int = 0) -> dict[str, Any]:
    try:
        return {"session_id": session_id, "events": SERVICE.get_events(session_id, start_frame=start_frame)}
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.get("/logs")
def get_logs(limit: int = 100) -> dict[str, Any]:
    return {"logs": SERVICE.get_logs(limit=limit)}


@app.post("/sessions/{session_id}/control")
def control_session(session_id: str, request: SessionControlRequest) -> dict[str, Any]:
    try:
        return SERVICE.control_session(session_id, request.action)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.post("/maintenance/cleanup")
def cleanup_sessions(keep_latest: int = 25) -> dict[str, Any]:
    return SERVICE.cleanup_sessions(keep_latest=keep_latest)


@app.post("/playback")
async def playback(request: BackendPlaybackRequest) -> StreamingResponse:
    try:
        session_id = SERVICE.prepare_session(
            request.csv_filename,
            max_seconds=request.max_seconds,
        )
    except Exception as exc:
        raise to_http_exception(exc) from exc

    async def event_stream():
        yield f"data: {json.dumps({'event': 'session-start', 'session_id': session_id})}\n\n"
        async for event in SERVICE.stream_session(session_id, delay_ms=request.delay_ms):
            yield f"data: {json.dumps(event)}\n\n"
        final_state = SERVICE.get_state(session_id)
        yield f"data: {json.dumps({'event': 'playback-complete', 'session_id': session_id, 'frames': final_state['frame_index'] + 1})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/trace/reset")
def reset_trace(request: TracePreviewRequest) -> dict[str, Any]:
    try:
        return SERVICE.preview_trace(request.csv_filename, max_seconds=request.max_seconds)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@app.websocket("/ws/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str, start_frame: int = 0, delay_ms: int = 0):
    await websocket.accept()
    try:
        initial = SERVICE.get_state(session_id)
        await websocket.send_json({"event": "state", "payload": initial})
        async for event in SERVICE.stream_session(session_id, delay_ms=delay_ms, start_frame=start_frame):
            await websocket.send_json({"event": "frame", "payload": event})
        await websocket.send_json({"event": "complete", "session_id": session_id})
    except SessionNotFoundError:
        await websocket.send_json({"event": "error", "detail": f"Session not found: {session_id}"})
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected for session %s", session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("simulation_engine:app", host="127.0.0.1", port=8000, reload=True)
