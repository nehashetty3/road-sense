from __future__ import annotations

import asyncio
import atexit
from contextlib import contextmanager
import json
import logging
import shutil
import sqlite3
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from experiments.common import (
    LOCS,
    construction_noise,
    load_geo,
    load_lnn,
    load_ode,
    load_swarm,
    load_towers,
    load_trace,
)
from layers.core import _heading_delta, run_all_layers
from layers.f1_lnn import lnn_predict_trace
from layers.f2_topology import compute_topology_fingerprint
from layers.f5_f6_f9 import SurpriseScoreModel, ode_reconstruction_error
from layers.tti_filter import compute_tti_filter


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_DEMO_DIR = ROOT / "csv_demo"
LOGGER = logging.getLogger("swarmmind.simulation")
SCHEMA_VERSION = 2


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
    "preset_name": "default",
}

KAN_WEIGHT_KEYS = set(DEFAULT_CONFIG["kan_weights"].keys())

CSV_ALIASES = {
    "normal.csv": ("la", "normal", "normal_la.csv"),
    "wrong_way.csv": ("la", "wrongway", "wrongway_la.csv"),
    "wrong_way_la.csv": ("la", "wrongway", "wrongway_la.csv"),
    "construction_detour.csv": ("la", "construction", "construction_la.csv"),
    "bike_path.csv": ("la", "normal", "normal_la.csv"),
    "normal_chennai.csv": ("chennai", "normal", "normal_chennai.csv"),
    "wrongway_chennai.csv": ("chennai", "wrongway", "wrongway_chennai.csv"),
    "normal_dc.csv": ("dc", "normal", "normal_dc.csv"),
    "wrongway_dc.csv": ("dc", "wrongway", "wrongway_dc.csv"),
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlaybackRequest(BaseModel):
    csv_filename: str = Field(..., examples=["wrong_way.csv"])
    delay_ms: int = Field(40, ge=0, le=1000)
    max_seconds: int | None = Field(90, ge=1, le=3600)
    preset_name: str | None = None


class ConfigUpdate(BaseModel):
    lnn_threshold: float | None = Field(None, ge=0.0, le=1.0)
    ode_threshold: float | None = Field(None, ge=0.0, le=1.0)
    tti_threshold: float | None = Field(None, ge=0.0, le=1.0)
    wrong_way_delta: float | None = Field(None, ge=45.0, le=180.0)
    min_wrong_steps: int | None = Field(None, ge=2, le=500)
    v2x_radius_km: float | None = Field(None, ge=0.1, le=20.0)
    kan_weights: dict[str, float] | None = None
    preset_name: str | None = None

    @model_validator(mode="after")
    def validate_weights(self):
        if self.kan_weights is not None:
            unknown = set(self.kan_weights) - KAN_WEIGHT_KEYS
            if unknown:
                raise ValueError(f"Unknown kan_weights keys: {sorted(unknown)}")
        return self


@dataclass
class TraceBundle:
    csv_name: str
    loc: str
    trace_type: str
    road_direction: float
    dataframe: pd.DataFrame
    geo: list[dict[str, Any]]
    towers: list[dict[str, Any]]
    swarm: list[dict[str, Any]]


class SimulationError(Exception):
    pass


class SessionNotFoundError(SimulationError):
    pass


class InvalidConfigError(SimulationError):
    pass


class InvalidTraceError(SimulationError):
    pass


class SimulationDB:
    def __init__(self, db_target: Path | str):
        self.db_target = str(db_target)
        self.backend = self._detect_backend(self.db_target)
        self.db_path = Path(db_target) if self.backend == "sqlite" else None
        self.db_url = self.db_target if self.backend == "postgres" else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @staticmethod
    def _detect_backend(target: str) -> str:
        parsed = urlparse(target)
        if parsed.scheme in {"postgres", "postgresql"}:
            return "postgres"
        return "sqlite"

    def _param(self, count: int) -> str:
        token = "%s" if self.backend == "postgres" else "?"
        return ", ".join([token] * count)

    def _connect(self):
        if self.backend == "postgres":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except Exception as exc:
                raise RuntimeError(f"psycopg is required for Postgres support: {exc}") from exc
            conn = psycopg.connect(self.db_url, row_factory=dict_row)
            conn.autocommit = False
            return conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=15000;")
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _fetch_value(self, conn, query: str, params: tuple[Any, ...] = ()) -> Any:
        row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        if isinstance(row, sqlite3.Row):
            return row[0]
        if isinstance(row, dict):
            return next(iter(row.values()))
        return row[0]

    def _init_db(self):
        with self._session() as conn:
            if self.backend == "postgres":
                ddl = """
                CREATE TABLE IF NOT EXISTS configs (
                    version BIGSERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    csv_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    frame_index INTEGER NOT NULL DEFAULT 0,
                    total_frames INTEGER NOT NULL DEFAULT 0,
                    current_time DOUBLE PRECISION NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL,
                    latest_state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    time DOUBLE PRECISION NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, frame_index)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    time DOUBLE PRECISION NOT NULL,
                    alert_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT,
                    action TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS request_logs (
                    id BIGSERIAL PRIMARY KEY,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    request_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS config_presets (
                    preset_name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );
                """
                for statement in [stmt.strip() for stmt in ddl.split(";") if stmt.strip()]:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO metadata(key, value_json) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value_json = EXCLUDED.value_json",
                    ("schema_version", json.dumps(SCHEMA_VERSION)),
                )
                if self._fetch_value(conn, "SELECT COUNT(*) FROM configs") == 0:
                    conn.execute(
                        "INSERT INTO configs(created_at, config_json) VALUES (%s, %s)",
                        (utcnow_iso(), json.dumps(DEFAULT_CONFIG)),
                    )
                if self._fetch_value(conn, "SELECT COUNT(*) FROM config_presets") == 0:
                    conn.execute(
                        "INSERT INTO config_presets(preset_name, created_at, config_json) VALUES (%s, %s, %s)",
                        ("default", utcnow_iso(), json.dumps(DEFAULT_CONFIG)),
                    )
            else:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS configs (
                        version INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        config_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        csv_filename TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        frame_index INTEGER NOT NULL DEFAULT 0,
                        total_frames INTEGER NOT NULL DEFAULT 0,
                        current_time REAL NOT NULL DEFAULT 0,
                        config_json TEXT NOT NULL,
                        latest_state_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        session_id TEXT NOT NULL,
                        frame_index INTEGER NOT NULL,
                        time REAL NOT NULL,
                        event_json TEXT NOT NULL,
                        PRIMARY KEY (session_id, frame_index)
                    );
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        time REAL NOT NULL,
                        alert_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        action TEXT NOT NULL,
                        detail_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS request_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        method TEXT NOT NULL,
                        path TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        request_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS config_presets (
                        preset_name TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        config_json TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key, value_json) VALUES (?, ?)",
                    ("schema_version", json.dumps(SCHEMA_VERSION)),
                )
                if self._fetch_value(conn, "SELECT COUNT(*) FROM configs") == 0:
                    conn.execute(
                        "INSERT INTO configs(created_at, config_json) VALUES (?, ?)",
                        (utcnow_iso(), json.dumps(DEFAULT_CONFIG)),
                    )
                if self._fetch_value(conn, "SELECT COUNT(*) FROM config_presets") == 0:
                    conn.execute(
                        "INSERT INTO config_presets(preset_name, created_at, config_json) VALUES (?, ?, ?)",
                        ("default", utcnow_iso(), json.dumps(DEFAULT_CONFIG)),
                    )
            conn.commit()

    def get_active_config(self) -> dict[str, Any]:
        with self._session() as conn:
            row = conn.execute("SELECT config_json FROM configs ORDER BY version DESC LIMIT 1").fetchone()
        return json.loads(row["config_json"])

    def list_config_versions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as conn:
            query = (
                "SELECT version, created_at, config_json FROM configs ORDER BY version DESC LIMIT %s"
                if self.backend == "postgres"
                else "SELECT version, created_at, config_json FROM configs ORDER BY version DESC LIMIT ?"
            )
            rows = conn.execute(query, (limit,)).fetchall()
        return [
            {
                "version": row["version"],
                "created_at": row["created_at"],
                "config": json.loads(row["config_json"]),
            }
            for row in rows
        ]

    def save_config(self, config: dict[str, Any]) -> dict[str, Any]:
        config = validate_config(config)
        with self._session() as conn:
            if self.backend == "postgres":
                cur = conn.execute(
                    "INSERT INTO configs(created_at, config_json) VALUES (%s, %s) RETURNING version",
                    (utcnow_iso(), json.dumps(config)),
                )
                version = cur.fetchone()["version"]
            else:
                cur = conn.execute(
                    "INSERT INTO configs(created_at, config_json) VALUES (?, ?)",
                    (utcnow_iso(), json.dumps(config)),
                )
                version = cur.lastrowid
            conn.commit()
        return {"version": version, "config": config}

    def get_config_version(self, version: int) -> dict[str, Any] | None:
        with self._session() as conn:
            query = "SELECT config_json FROM configs WHERE version=%s" if self.backend == "postgres" else "SELECT config_json FROM configs WHERE version=?"
            row = conn.execute(query, (version,)).fetchone()
        return json.loads(row["config_json"]) if row else None

    def save_preset(self, preset_name: str, config: dict[str, Any]):
        with self._session() as conn:
            payload = (preset_name, utcnow_iso(), json.dumps(validate_config(config)))
            if self.backend == "postgres":
                conn.execute(
                    "INSERT INTO config_presets(preset_name, created_at, config_json) VALUES (%s, %s, %s) "
                    "ON CONFLICT (preset_name) DO UPDATE SET created_at = EXCLUDED.created_at, config_json = EXCLUDED.config_json",
                    payload,
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO config_presets(preset_name, created_at, config_json) VALUES (?, ?, ?)",
                    payload,
                )
            conn.commit()

    def list_presets(self) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT preset_name, created_at, config_json FROM config_presets ORDER BY preset_name ASC"
            ).fetchall()
        return [
            {"preset_name": row["preset_name"], "created_at": row["created_at"], "config": json.loads(row["config_json"])}
            for row in rows
        ]

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT session_id, csv_filename, status, created_at, updated_at, frame_index, total_frames, current_time "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_session_id(self) -> str | None:
        with self._session() as conn:
            row = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT 1").fetchone()
        return row["session_id"] if row else None

    def create_session(self, session_id: str, csv_filename: str, config: dict[str, Any], initial_state: dict[str, Any], total_frames: int):
        now = utcnow_iso()
        with self._session() as conn:
            payload = (
                session_id,
                csv_filename,
                "prepared",
                now,
                now,
                0,
                total_frames,
                float(initial_state.get("time", 0.0)),
                json.dumps(config),
                json.dumps(initial_state),
            )
            query = (
                """
                INSERT INTO sessions(session_id, csv_filename, status, created_at, updated_at, frame_index, total_frames, current_time, config_json, latest_state_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                if self.backend == "postgres"
                else """
                INSERT INTO sessions(session_id, csv_filename, status, created_at, updated_at, frame_index, total_frames, current_time, config_json, latest_state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )
            conn.execute(query, payload)
            conn.commit()

    def update_session_state(self, session_id: str, *, status: str, frame_index: int, total_frames: int, current_time: float, latest_state: dict[str, Any]):
        with self._session() as conn:
            payload = (status, utcnow_iso(), frame_index, total_frames, current_time, json.dumps(latest_state), session_id)
            query = (
                """
                UPDATE sessions
                SET status=%s, updated_at=%s, frame_index=%s, total_frames=%s, current_time=%s, latest_state_json=%s
                WHERE session_id=%s
                """
                if self.backend == "postgres"
                else """
                UPDATE sessions
                SET status=?, updated_at=?, frame_index=?, total_frames=?, current_time=?, latest_state_json=?
                WHERE session_id=?
                """
            )
            conn.execute(query, payload)
            conn.commit()

    def update_session_status(self, session_id: str, status: str):
        with self._session() as conn:
            query = "UPDATE sessions SET status=%s, updated_at=%s WHERE session_id=%s" if self.backend == "postgres" else "UPDATE sessions SET status=?, updated_at=? WHERE session_id=?"
            conn.execute(query, (status, utcnow_iso(), session_id))
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._session() as conn:
            query = "SELECT * FROM sessions WHERE session_id=%s" if self.backend == "postgres" else "SELECT * FROM sessions WHERE session_id=?"
            row = conn.execute(query, (session_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["config"] = json.loads(out.pop("config_json"))
        out["latest_state"] = json.loads(out.pop("latest_state_json"))
        return out

    def store_events(self, session_id: str, events: list[dict[str, Any]]):
        with self._session() as conn:
            event_rows = [
                (session_id, idx, float(event["time"]), json.dumps(event))
                for idx, event in enumerate(events)
            ]
            alert_rows = [
                (session_id, float(event["time"]), json.dumps(event["alert"]))
                for event in events
                if event.get("alert")
            ]
            if self.backend == "postgres":
                conn.executemany(
                    "INSERT INTO events(session_id, frame_index, time, event_json) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (session_id, frame_index) DO UPDATE SET time = EXCLUDED.time, event_json = EXCLUDED.event_json",
                    event_rows,
                )
                if alert_rows:
                    conn.executemany(
                        "INSERT INTO alerts(session_id, time, alert_json) VALUES (%s, %s, %s)",
                        alert_rows,
                    )
            else:
                conn.executemany(
                    "INSERT OR REPLACE INTO events(session_id, frame_index, time, event_json) VALUES (?, ?, ?, ?)",
                    event_rows,
                )
                if alert_rows:
                    conn.executemany(
                        "INSERT INTO alerts(session_id, time, alert_json) VALUES (?, ?, ?)",
                        alert_rows,
                    )
            conn.commit()

    def get_events(self, session_id: str, start_frame: int = 0) -> list[dict[str, Any]]:
        with self._session() as conn:
            query = (
                "SELECT event_json FROM events WHERE session_id=%s AND frame_index>=%s ORDER BY frame_index ASC"
                if self.backend == "postgres"
                else "SELECT event_json FROM events WHERE session_id=? AND frame_index>=? ORDER BY frame_index ASC"
            )
            rows = conn.execute(query, (session_id, start_frame)).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def get_alerts(self, session_id: str) -> list[dict[str, Any]]:
        with self._session() as conn:
            query = "SELECT alert_json FROM alerts WHERE session_id=%s ORDER BY time ASC" if self.backend == "postgres" else "SELECT alert_json FROM alerts WHERE session_id=? ORDER BY time ASC"
            rows = conn.execute(query, (session_id,)).fetchall()
        return [json.loads(row["alert_json"]) for row in rows]

    def add_audit_log(self, action: str, detail: dict[str, Any], session_id: str | None = None):
        with self._session() as conn:
            query = "INSERT INTO audit_logs(session_id, action, detail_json, created_at) VALUES (%s, %s, %s, %s)" if self.backend == "postgres" else "INSERT INTO audit_logs(session_id, action, detail_json, created_at) VALUES (?, ?, ?, ?)"
            conn.execute(query, (session_id, action, json.dumps(detail), utcnow_iso()))
            conn.commit()

    def list_audit_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as conn:
            query = (
                "SELECT session_id, action, detail_json, created_at FROM audit_logs ORDER BY id DESC LIMIT %s"
                if self.backend == "postgres"
                else "SELECT session_id, action, detail_json, created_at FROM audit_logs ORDER BY id DESC LIMIT ?"
            )
            rows = conn.execute(query, (limit,)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json"))
            out.append(item)
        return out

    def add_request_log(self, method: str, path: str, status_code: int, request_id: str | None = None):
        with self._session() as conn:
            query = "INSERT INTO request_logs(method, path, status_code, request_id, created_at) VALUES (%s, %s, %s, %s, %s)" if self.backend == "postgres" else "INSERT INTO request_logs(method, path, status_code, request_id, created_at) VALUES (?, ?, ?, ?, ?)"
            conn.execute(query, (method, path, status_code, request_id, utcnow_iso()))
            conn.commit()

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._session() as conn:
            session_rows = conn.execute("SELECT status FROM sessions").fetchall()
            alert_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            request_count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
            config_versions = conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0]
        status_counts = Counter(row["status"] for row in session_rows)
        return {
            "schema_version": SCHEMA_VERSION,
            "session_counts": dict(status_counts),
            "total_sessions": len(session_rows),
            "total_alerts": int(alert_count),
            "total_requests": int(request_count),
            "config_versions": int(config_versions),
        }

    def cleanup_old_sessions(self, keep_latest: int = 25) -> dict[str, int]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            to_delete = [row["session_id"] for row in rows[keep_latest:]]
            if not to_delete:
                return {"deleted_sessions": 0}
            if self.backend == "postgres":
                conn.executemany("DELETE FROM events WHERE session_id=%s", [(sid,) for sid in to_delete])
                conn.executemany("DELETE FROM alerts WHERE session_id=%s", [(sid,) for sid in to_delete])
                conn.executemany("DELETE FROM audit_logs WHERE session_id=%s", [(sid,) for sid in to_delete])
                conn.executemany("DELETE FROM sessions WHERE session_id=%s", [(sid,) for sid in to_delete])
            else:
                conn.executemany("DELETE FROM events WHERE session_id=?", [(sid,) for sid in to_delete])
                conn.executemany("DELETE FROM alerts WHERE session_id=?", [(sid,) for sid in to_delete])
                conn.executemany("DELETE FROM audit_logs WHERE session_id=?", [(sid,) for sid in to_delete])
                conn.executemany("DELETE FROM sessions WHERE session_id=?", [(sid,) for sid in to_delete])
            conn.commit()
        return {"deleted_sessions": len(to_delete)}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged.update({k: v for k, v in config.items() if k != "kan_weights"})
    merged["kan_weights"].update(config.get("kan_weights", {}))

    if merged["lnn_threshold"] > merged["tti_threshold"]:
        raise InvalidConfigError("lnn_threshold must not exceed tti_threshold")
    if merged["ode_threshold"] > 1.0:
        raise InvalidConfigError("ode_threshold must be <= 1.0")
    if merged["min_wrong_steps"] < 2:
        raise InvalidConfigError("min_wrong_steps must be at least 2")
    preset_name = merged.get("preset_name")
    if preset_name is not None:
        if not isinstance(preset_name, str) or not preset_name.strip():
            raise InvalidConfigError("preset_name must be a non-empty string")
        merged["preset_name"] = preset_name.strip()
    return merged


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


def resolve_csv_alias(csv_filename: str) -> tuple[str, str, str]:
    ensure_csv_demo()
    clean = Path(csv_filename).name
    if clean in CSV_ALIASES:
        return CSV_ALIASES[clean]
    lower = clean.lower()
    if "chennai" in lower:
        loc = "chennai"
    elif "dc" in lower:
        loc = "dc"
    else:
        loc = "la"
    trace_type = "wrongway" if "wrong" in lower else "construction" if "construction" in lower or "detour" in lower else "fault" if "fault" in lower else "drunk" if "drunk" in lower else "normal"
    return loc, trace_type, clean


def resolve_csv_path(csv_filename: str) -> Path:
    clean = Path(csv_filename).name
    candidates = [CSV_DEMO_DIR / clean, DATA_DIR / clean]
    loc, trace_type, canonical = resolve_csv_alias(clean)
    candidates.extend([CSV_DEMO_DIR / canonical, DATA_DIR / canonical])
    for candidate in candidates:
        if candidate.exists() and candidate.suffix == ".csv":
            return candidate
    raise InvalidTraceError(f"CSV not found: {csv_filename}")


def file_provenance() -> dict[str, Any]:
    def stamp(path: Path):
        return {"path": str(path.name), "mtime": path.stat().st_mtime} if path.exists() else None

    return {
        "lnn_weights": stamp(DATA_DIR / "lnn_weights.pt"),
        "ode_weights": stamp(DATA_DIR / "ode_weights.pt"),
        "tti_weights": stamp(DATA_DIR / "tti_filter_weights.json"),
        "ngsim_csv": stamp(ROOT / "external_data" / "ngsim" / "ngsim_full.csv"),
    }


class SwarmMindRuntimeEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = validate_config(config)

    def load_trace_bundle(self, csv_filename: str) -> TraceBundle:
        requested_name = Path(csv_filename).name
        loc, trace_type, canonical = resolve_csv_alias(csv_filename)
        path = resolve_csv_path(canonical)
        if requested_name == "bike_path.csv":
            df = pd.read_csv(CSV_DEMO_DIR / requested_name)
        elif path.parent == DATA_DIR and path.name.startswith(trace_type):
            df = load_trace(loc, trace_type)
        elif path.parent == CSV_DEMO_DIR and canonical in CSV_ALIASES:
            df = load_trace(loc, trace_type)
        else:
            df = pd.read_csv(path)
        road_direction = 130.0 if loc == "us101" else LOCS.get(loc, 350.0)
        return TraceBundle(
            csv_name=requested_name,
            loc=loc,
            trace_type=trace_type,
            road_direction=road_direction,
            dataframe=df.sort_values("time").reset_index(drop=True),
            geo=load_geo(loc),
            towers=load_towers(loc),
            swarm=load_swarm(loc),
        )

    def prepare_session_frames(self, csv_filename: str, max_seconds: int | None = None) -> list[dict[str, Any]]:
        bundle = self.load_trace_bundle(csv_filename)
        df = bundle.dataframe
        if max_seconds is not None:
            df = df[df["time"] <= max_seconds].reset_index(drop=True)
        pipeline = run_all_layers(df, bundle.geo, bundle.towers, bundle.swarm, bundle.road_direction, 75)
        lnn_model = load_lnn()
        lnn_res = lnn_predict_trace(lnn_model, df, threshold=self.config["lnn_threshold"], road_direction=bundle.road_direction)
        ode_model, mins, ranges = load_ode()
        ode_res = ode_reconstruction_error(ode_model, mins, ranges, df)
        topo_res = compute_topology_fingerprint(df)
        normal_reference = load_trace(bundle.loc, "normal") if bundle.trace_type != "normal" else df
        surprise = SurpriseScoreModel()
        surprise.fit(normal_reference)
        surprise_res = surprise.score_trace(df)
        tti_res = compute_tti_filter(
            ode_res["max_mse"],
            topo_res["max_persistence"],
            surprise_res["max_accumulated"],
            construction_noise(df, bundle.road_direction),
        )
        return self._build_frames(bundle, pipeline, lnn_res, ode_res, topo_res, surprise_res, tti_res)

    def _build_frames(
        self,
        bundle: TraceBundle,
        pipeline: dict[str, Any],
        lnn_res: dict[str, Any],
        ode_res: dict[str, Any],
        topo_res: dict[str, Any],
        surprise_res: dict[str, Any],
        tti_res: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results = pipeline["layer1"]["results"]
        alert_by_time = {}
        for alert in pipeline["layer2"]["alerts"]:
            alert_by_time.setdefault(int(alert["time"]), []).append(alert)
        coverage = pipeline["layer9"].get("coverage_per_point", [])
        frames = []
        seen_alerts = 0
        for idx, result in enumerate(results):
            time_key = int(result["t"])
            alerts_now = alert_by_time.get(time_key, [])
            seen_alerts += len(alerts_now)
            top_alert = alerts_now[0] if alerts_now else None
            status = "WRONG-WAY DETECTED" if top_alert else "CAUTION" if result["wrong_way"] else "NORMAL"
            frame = {
                "time": round(float(result["t"]), 2),
                "vehicles": self._build_scene_vehicles(bundle, idx, status),
                "alert": self._format_alert(top_alert, tti_res) if top_alert else None,
                "collision": self._collision_overlay(bundle, idx, result, pipeline, top_alert),
                "layers": self._frame_layers(bundle, result, idx, pipeline, lnn_res, ode_res, topo_res, surprise_res, tti_res, coverage, top_alert, seen_alerts),
                "status": status,
                "session_meta": {
                    "loc": bundle.loc,
                    "trace_type": bundle.trace_type,
                    "csv_name": bundle.csv_name,
                },
            }
            frames.append(frame)
        return frames

    def _format_alert(self, alert: dict[str, Any] | None, tti_res: dict[str, Any]) -> dict[str, Any] | None:
        if not alert:
            return None
        return {
            "time": round(float(alert["time"]), 2),
            "type": alert["severity"],
            "lat": alert["lat"],
            "lon": alert["lon"],
            "confidence": round(float(max(alert["confidence"], tti_res["confidence"])), 3),
            "location": f"{alert['lat']:.5f}, {alert['lon']:.5f}",
            "channels": alert["channels"],
            "v2x_radius_km": self.config["v2x_radius_km"],
        }

    def _collision_overlay(self, bundle: TraceBundle, idx: int, result: dict[str, Any], pipeline: dict[str, Any], top_alert: dict[str, Any] | None) -> dict[str, Any]:
        row = bundle.dataframe.iloc[min(idx, len(bundle.dataframe) - 1)]
        t = float(row["time"])
        speed = float(row["speed"])
        risk_score = float(pipeline["layer5"]["risk_score"])
        impact_count = len(pipeline["layer5"].get("predicted_impacts", []))
        construction_like = bundle.trace_type == "construction"
        ego_wrong = bundle.trace_type == "wrongway"

        def normal_progress(seed: float, velocity: float) -> float:
            return (seed + (t * velocity)) % 118.0 - 9.0

        def inverted_progress(seed: float, velocity: float) -> float:
            return 100.0 - ((seed + (t * velocity)) % 118.0 - 9.0)

        lane = 1 if ego_wrong else 1
        if construction_like:
            lane = 1 if int(t // 8) % 2 else 0
        progress = inverted_progress(12.0, max(0.35, speed / 115.0)) if ego_wrong else normal_progress(18.0, max(0.35, speed / 130.0))
        active = bool(top_alert) or risk_score >= 0.45 or impact_count > 0
        return {
            "active": active,
            "lane": lane,
            "progress": round(float(progress), 2),
            "risk_score": round(risk_score, 3),
            "impact_count": impact_count,
        }

    def _frame_layers(self, bundle, result, idx, pipeline, lnn_res, ode_res, topo_res, surprise_res, tti_res, coverage, top_alert, seen_alerts):
        construction_like = bundle.trace_type == "construction"
        bike_like = "bike" in bundle.csv_name.lower()
        layer_values = [
            ("L1 GPS Consensus", max(result["confidence"], min(1.0, result["road_delta"] / 180.0)), "active" if result["wrong_way"] else "idle"),
            ("L2 Alert Routing", top_alert["confidence"] if top_alert else 0.0, "active" if top_alert else "idle"),
            ("L3 FFT Pattern", min(1.0, pipeline["layer3"]["dominant_amp"] / 15.0), "active" if pipeline["layer3"]["attack_type"] != "normal" else "idle"),
            ("L4 rPPG Visual Only", 0.0, "suppressed"),
            ("L5 Collision Cloud", min(1.0, pipeline["layer5"]["risk_score"]), "active" if pipeline["layer5"]["predicted_impacts"] else "idle"),
            ("L6 Micromobility Guard", min(1.0, pipeline["layer6"]["suppression_rate"]), "suppressed" if bike_like or pipeline["layer6"]["suppressed_count"] else "idle"),
            ("L7 Work-Zone Guard", min(1.0, pipeline["layer7"]["fp_rate_reduction"]), "suppressed" if construction_like and pipeline["layer7"]["downgraded_count"] else "active" if construction_like else "idle"),
            ("L8 V2X Ripple", min(1.0, seen_alerts / 3.0 if seen_alerts else 0.0), "active" if seen_alerts else "idle"),
            ("L9 Connectivity", round(float(coverage[min(idx, len(coverage) - 1)]) if coverage else 0.78, 3), "active"),
        ]
        return [
            {"layer": layer, "confidence": round(float(conf), 3), "status": "failed" if status == "active" and conf < 0.15 else status}
            for layer, conf, status in layer_values
        ]

    def _build_scene_vehicles(self, bundle: TraceBundle, idx: int, status: str) -> list[dict[str, Any]]:
        row = bundle.dataframe.iloc[min(idx, len(bundle.dataframe) - 1)]
        t = float(row["time"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        heading = float(row["heading"])
        speed = float(row["speed"])
        lower = bundle.csv_name.lower()
        lane_centers = {0: 28.0, 1: 42.0, 2: 58.0, 3: 72.0}

        def normal_progress(seed: float, velocity: float) -> float:
            return (seed + (t * velocity)) % 118.0 - 9.0

        def inverted_progress(seed: float, velocity: float) -> float:
            return 100.0 - ((seed + (t * velocity)) % 118.0 - 9.0)

        ego_wrong = bundle.trace_type == "wrongway"
        construction_like = bundle.trace_type == "construction"
        ego_lane = 1 if ego_wrong else 1
        if construction_like:
            ego_lane = 1 if int(t // 8) % 2 == 0 else 0
        ego_progress = inverted_progress(12.0, max(0.35, speed / 115.0)) if ego_wrong else normal_progress(18.0, max(0.35, speed / 130.0))
        vehicles = [
            {
                "id": "ego-001",
                "lat": lat,
                "lon": lon,
                "heading": round(heading, 2),
                "speed": round(speed, 2),
                "behavior": "wrong-way-confirmed" if status == "WRONG-WAY DETECTED" else "construction-detour" if construction_like else "counter-flow-caution" if ego_wrong else "normal",
                "lane": ego_lane,
                "progress": round(float(ego_progress), 2),
                "direction": "southbound" if ego_wrong else "northbound",
                "is_alert_source": status != "NORMAL",
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
            sprite_override = sprite
            if construction_like and direction == "northbound" and lane == 1 and 25.0 < t < 58.0:
                lane_override = 0
                sim_behavior = "detour"
            if "bike" in lower and vid == "north-03":
                lane_override = 0
                sim_behavior = "bike-path-nearby"
                sprite_override = "bike"
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
                    "sprite": sprite_override,
                }
            )
        return vehicles


class SimulationService:
    def __init__(self, db_path: Path):
        self.db = SimulationDB(db_path)
        self._session_lock = threading.Lock()
        ensure_csv_demo()
        atexit.register(self._shutdown)

    def _shutdown(self):
        # Placeholder for future connection pools / background workers.
        return None

    def bootstrap_state(self) -> dict[str, Any]:
        latest = self.db.get_latest_session_id()
        if latest:
            return self.get_state(latest)
        session_id = self.prepare_session("normal.csv")
        return self.get_state(session_id)

    def get_config(self) -> dict[str, Any]:
        return self.db.get_active_config()

    def get_config_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.list_config_versions(limit=limit)

    def update_config(self, update: dict[str, Any]) -> dict[str, Any]:
        base = self.get_config()
        merged = json.loads(json.dumps(base))
        for key, value in update.items():
            if key == "kan_weights" and value:
                merged["kan_weights"].update(value)
            else:
                merged[key] = value
        saved = self.db.save_config(merged)
        self.db.add_audit_log("config_updated", {"version": saved["version"], "config": saved["config"]})
        return saved

    def rollback_config(self, version: int) -> dict[str, Any]:
        config = self.db.get_config_version(version)
        if not config:
            raise InvalidConfigError(f"Unknown config version: {version}")
        saved = self.db.save_config(config)
        self.db.add_audit_log("config_rolled_back", {"from_version": version, "new_version": saved["version"]})
        return saved

    def save_preset(self, preset_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if not preset_name.strip():
            raise InvalidConfigError("preset_name must not be empty")
        active = validate_config(config or self.get_config())
        normalized_name = preset_name.strip()
        active["preset_name"] = normalized_name
        self.db.save_preset(normalized_name, active)
        self.db.add_audit_log("preset_saved", {"preset_name": normalized_name})
        return {"preset_name": normalized_name, "config": active}

    def list_presets(self) -> list[dict[str, Any]]:
        return self.db.list_presets()

    def _route_metadata(self, engine: SwarmMindRuntimeEngine, csv_filename: str, max_seconds: int | None) -> dict[str, Any]:
        bundle = engine.load_trace_bundle(csv_filename)
        df = bundle.dataframe
        if max_seconds is not None:
            df = df[df["time"] <= max_seconds].reset_index(drop=True)
        if df.empty:
            return {"route_geometry": [], "map_center": None, "map_bounds": None}
        stride = max(1, len(df) // 48)
        sampled = df.iloc[::stride][["lon", "lat"]].dropna()
        if sampled.empty or sampled.iloc[-1].tolist() != [float(df.iloc[-1]["lon"]), float(df.iloc[-1]["lat"])]:
            sampled = pd.concat([sampled, df.iloc[[-1]][["lon", "lat"]]], ignore_index=True)
        route_geometry = [
            [round(float(row["lon"]), 6), round(float(row["lat"]), 6)]
            for _, row in sampled.iterrows()
        ]
        lons = [pt[0] for pt in route_geometry]
        lats = [pt[1] for pt in route_geometry]
        center = [round(sum(lons) / len(lons), 6), round(sum(lats) / len(lats), 6)]
        bounds = [
            [round(min(lons), 6), round(min(lats), 6)],
            [round(max(lons), 6), round(max(lats), 6)],
        ]
        return {
            "route_geometry": route_geometry,
            "map_center": center,
            "map_bounds": bounds,
        }

    def prepare_session(self, csv_filename: str, max_seconds: int | None = 90, config: dict[str, Any] | None = None) -> str:
        active_config = validate_config(config or self.get_config())
        engine = SwarmMindRuntimeEngine(active_config)
        frames = engine.prepare_session_frames(csv_filename, max_seconds=max_seconds)
        if not frames:
            raise InvalidTraceError(f"No frames produced for {csv_filename}")
        route_meta = self._route_metadata(engine, csv_filename, max_seconds)
        session_id = uuid.uuid4().hex
        initial = {
            "session_id": session_id,
            "time": frames[0]["time"],
            "status": frames[0]["status"],
            "vehicles": frames[0]["vehicles"],
            "alerts": [],
            "collision": frames[0].get("collision"),
            "layers": frames[0]["layers"],
            "active_csv": csv_filename,
            "playback_running": False,
            "frame_index": 0,
            "total_frames": len(frames),
            "provenance": file_provenance(),
            **route_meta,
        }
        self.db.create_session(session_id, csv_filename, active_config, initial, len(frames))
        self.db.store_events(session_id, frames)
        self.db.add_audit_log(
            "session_prepared",
            {"csv_filename": csv_filename, "total_frames": len(frames), "provenance": file_provenance()},
            session_id=session_id,
        )
        return session_id

    def preview_trace(self, csv_filename: str, max_seconds: int | None = 90) -> dict[str, Any]:
        session_id = self.prepare_session(csv_filename, max_seconds=max_seconds)
        state = self.get_state(session_id)
        state["playback_running"] = False
        return state

    def list_csvs(self) -> list[str]:
        ensure_csv_demo()
        return sorted(path.name for path in CSV_DEMO_DIR.glob("*.csv"))

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.db.list_sessions()

    def get_state(self, session_id: str | None = None) -> dict[str, Any]:
        target = session_id or self.db.get_latest_session_id()
        if not target:
            return self.bootstrap_state()
        session = self.db.get_session(target)
        if not session:
            raise SessionNotFoundError(f"Session not found: {target}")
        latest_state = session["latest_state"]
        latest_state["session_id"] = session["session_id"]
        latest_state["config"] = session["config"]
        latest_state["total_frames"] = session["total_frames"]
        latest_state["current_time"] = session["current_time"]
        return latest_state

    def get_alerts(self, session_id: str) -> list[dict[str, Any]]:
        return self.db.get_alerts(session_id)

    def get_events(self, session_id: str, start_frame: int = 0) -> list[dict[str, Any]]:
        session = self.db.get_session(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return self.db.get_events(session_id, start_frame=start_frame)

    def get_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.list_audit_logs(limit=limit)

    def get_metrics(self) -> dict[str, Any]:
        return self.db.metrics_snapshot()

    def cleanup_sessions(self, keep_latest: int = 25) -> dict[str, int]:
        result = self.db.cleanup_old_sessions(keep_latest=keep_latest)
        self.db.add_audit_log("cleanup_sessions", result)
        return result

    def control_session(self, session_id: str, action: str) -> dict[str, Any]:
        session = self.db.get_session(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        if action not in {"pause", "resume", "cancel"}:
            raise SimulationError(f"Unsupported session action: {action}")
        status = {"pause": "paused", "resume": "playing", "cancel": "cancelled"}[action]
        self.db.update_session_status(session_id, status)
        self.db.add_audit_log("session_control", {"action": action}, session_id=session_id)
        return {"session_id": session_id, "status": status}

    async def stream_session(self, session_id: str, delay_ms: int = 40, start_frame: int = 0):
        events = self.db.get_events(session_id, start_frame=start_frame)
        session = self.db.get_session(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        alerts_seen: list[dict[str, Any]] = []
        self.db.update_session_state(
            session_id,
            status="playing",
            frame_index=start_frame,
            total_frames=session["total_frames"],
            current_time=session["current_time"],
            latest_state=session["latest_state"],
        )
        try:
            for idx, event in enumerate(events, start=start_frame):
                while True:
                    current = self.db.get_session(session_id)
                    if not current:
                        raise SessionNotFoundError(f"Session not found: {session_id}")
                    if current["status"] == "cancelled":
                        self.db.add_audit_log("session_cancelled", {"frame_index": idx}, session_id=session_id)
                        return
                    if current["status"] != "paused":
                        break
                    await asyncio.sleep(0.1)
                if event.get("alert"):
                    alerts_seen.append(event["alert"])
                latest_state = {
                    "session_id": session_id,
                    "time": event["time"],
                    "status": event["status"],
                    "vehicles": event["vehicles"],
                    "alerts": alerts_seen[-100:],
                    "collision": event.get("collision"),
                    "layers": event["layers"],
                    "active_csv": session["csv_filename"],
                    "playback_running": True,
                    "frame_index": idx,
                    "provenance": session["latest_state"].get("provenance", file_provenance()),
                    "route_geometry": session["latest_state"].get("route_geometry", []),
                    "map_center": session["latest_state"].get("map_center"),
                    "map_bounds": session["latest_state"].get("map_bounds"),
                }
                self.db.update_session_state(
                    session_id,
                    status="playing",
                    frame_index=idx,
                    total_frames=session["total_frames"],
                    current_time=event["time"],
                    latest_state=latest_state,
                )
                enriched = dict(event)
                enriched["session_id"] = session_id
                yield enriched
                if delay_ms:
                    await asyncio.sleep(delay_ms / 1000.0)
            final_state = self.get_state(session_id)
            final_state["playback_running"] = False
            self.db.update_session_state(
                session_id,
                status="completed",
                frame_index=final_state["frame_index"],
                total_frames=session["total_frames"],
                current_time=final_state["time"],
                latest_state=final_state,
            )
            self.db.add_audit_log("session_completed", {"frames": session["total_frames"]}, session_id=session_id)
        except Exception as exc:
            self.db.add_audit_log("session_failed", {"error": str(exc)}, session_id=session_id)
            raise
