from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import importlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


def _encode_jwt(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def _b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    header_b64 = _b64(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


class SimulationBackendTests(unittest.TestCase):
    def test_database_backend_detection(self):
        from simulation_backend import SimulationDB

        self.assertEqual(SimulationDB._detect_backend("/tmp/swarmmind-detect.db"), "sqlite")
        self.assertEqual(SimulationDB._detect_backend("postgresql://demo:demo@localhost:5432/swarmmind"), "postgres")

    def test_prepare_session_is_deterministic(self):
        from simulation_backend import SimulationService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(Path(tmpdir) / "sim.db")
            session_one = service.prepare_session("wrong_way.csv", max_seconds=12)
            session_two = service.prepare_session("wrong_way.csv", max_seconds=12)
            events_one = service.db.get_events(session_one)
            events_two = service.db.get_events(session_two)
            self.assertGreater(len(events_one), 0)
            self.assertEqual(
                [{k: event[k] for k in ("time", "status", "alert", "layers")} for event in events_one],
                [{k: event[k] for k in ("time", "status", "alert", "layers")} for event in events_two],
            )

    def test_stream_session_updates_persisted_state(self):
        from simulation_backend import SimulationService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(Path(tmpdir) / "sim.db")
            session_id = service.prepare_session("wrong_way.csv", max_seconds=10)

            async def collect():
                out = []
                async for frame in service.stream_session(session_id, delay_ms=0):
                    out.append(frame)
                return out

            frames = asyncio.run(collect())
            state = service.get_state(session_id)
            self.assertGreater(len(frames), 0)
            self.assertEqual(state["status"], frames[-1]["status"])
            self.assertEqual(state["frame_index"], len(frames) - 1)

    def test_fastapi_routes_expose_sessions_and_logs(self):
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        old_env = os.environ.get("SWARMMIND_SIM_DB")
        os.environ["SWARMMIND_SIM_DB"] = db_path
        try:
            import simulation_engine

            simulation_engine = importlib.reload(simulation_engine)
            client = TestClient(simulation_engine.app)
            with client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/readiness").status_code, 200)
                self.assertEqual(client.get("/dependencies").status_code, 200)
                self.assertEqual(client.get("/state").status_code, 200)
                config = client.get("/config").json()
                self.assertIn("lnn_threshold", config)

                playback = client.post("/playback", json={"csv_filename": "wrong_way.csv", "delay_ms": 0, "max_seconds": 8})
                self.assertEqual(playback.status_code, 200)
                self.assertIn("session-start", playback.text)

                sessions = client.get("/sessions").json()["sessions"]
                self.assertGreater(len(sessions), 0)
                session_id = sessions[0]["session_id"]
                alerts = client.get(f"/alerts/{session_id}").json()["alerts"]
                events = client.get(f"/events/{session_id}").json()["events"]
                logs = client.get("/logs").json()["logs"]
                metrics = client.get("/metrics").json()
                config_history = client.get("/config/history").json()["versions"]
                readiness = client.get("/readiness").json()
                dependencies = client.get("/dependencies").json()
                self.assertIsInstance(alerts, list)
                self.assertGreater(len(events), 0)
                self.assertGreater(len(logs), 0)
                self.assertGreaterEqual(metrics["total_sessions"], 1)
                self.assertGreaterEqual(len(config_history), 1)
                self.assertIn("database", dependencies)
                self.assertIn("ready", readiness)
        finally:
            if old_env is None:
                os.environ.pop("SWARMMIND_SIM_DB", None)
            else:
                os.environ["SWARMMIND_SIM_DB"] = old_env
            Path(db_path).unlink(missing_ok=True)

    def test_api_key_auth_blocks_non_local_without_key(self):
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        old_env = {name: os.environ.get(name) for name in ("SWARMMIND_SIM_DB", "SWARMMIND_AUTH_MODE", "SWARMMIND_API_KEY", "SWARMMIND_ALLOW_INSECURE_LOCALHOST")}
        os.environ["SWARMMIND_SIM_DB"] = db_path
        os.environ["SWARMMIND_AUTH_MODE"] = "api_key"
        os.environ["SWARMMIND_API_KEY"] = "secret-demo-key"
        os.environ["SWARMMIND_ALLOW_INSECURE_LOCALHOST"] = "false"
        try:
            import simulation_engine

            simulation_engine = importlib.reload(simulation_engine)
            client = TestClient(simulation_engine.app)
            with client:
                unauthorized = client.get("/state")
                self.assertEqual(unauthorized.status_code, 401)
                authorized = client.get("/state", headers={"x-api-key": "secret-demo-key"})
                self.assertEqual(authorized.status_code, 200)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            Path(db_path).unlink(missing_ok=True)

    def test_bearer_auth_allows_signed_token(self):
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        keys = (
            "SWARMMIND_SIM_DB",
            "SWARMMIND_AUTH_MODE",
            "SWARMMIND_BEARER_SECRET",
            "SWARMMIND_BEARER_ISSUER",
            "SWARMMIND_BEARER_AUDIENCE",
            "SWARMMIND_ALLOW_INSECURE_LOCALHOST",
        )
        old_env = {name: os.environ.get(name) for name in keys}
        os.environ["SWARMMIND_SIM_DB"] = db_path
        os.environ["SWARMMIND_AUTH_MODE"] = "bearer"
        os.environ["SWARMMIND_BEARER_SECRET"] = "super-secret"
        os.environ["SWARMMIND_BEARER_ISSUER"] = "https://swarmmind.test"
        os.environ["SWARMMIND_BEARER_AUDIENCE"] = "swarmmind-api"
        os.environ["SWARMMIND_ALLOW_INSECURE_LOCALHOST"] = "false"
        try:
            import simulation_engine

            simulation_engine = importlib.reload(simulation_engine)
            client = TestClient(simulation_engine.app)
            token = _encode_jwt(
                {
                    "sub": "tester",
                    "iss": "https://swarmmind.test",
                    "aud": "swarmmind-api",
                    "exp": int(time.time()) + 300,
                },
                "super-secret",
            )
            with client:
                unauthorized = client.get("/state")
                self.assertEqual(unauthorized.status_code, 401)
                authorized = client.get("/state", headers={"authorization": f"Bearer {token}"})
                self.assertEqual(authorized.status_code, 200)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            Path(db_path).unlink(missing_ok=True)

    def test_presets_rollback_and_cleanup_controls(self):
        from simulation_backend import SimulationService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(Path(tmpdir) / "sim.db")
            original = service.get_config()
            updated = service.update_config({"lnn_threshold": 0.25})
            self.assertNotEqual(original["lnn_threshold"], updated["config"]["lnn_threshold"])

            rollback = service.rollback_config(1)
            self.assertEqual(rollback["config"]["lnn_threshold"], original["lnn_threshold"])

            preset = service.save_preset("night_run")
            preset_names = [item["preset_name"] for item in service.list_presets()]
            self.assertEqual(preset["preset_name"], "night_run")
            self.assertIn("night_run", preset_names)

            session_ids = [service.prepare_session("normal.csv", max_seconds=5) for _ in range(3)]
            cleanup = service.cleanup_sessions(keep_latest=1)
            self.assertEqual(cleanup["deleted_sessions"], 2)
            remaining = service.list_sessions()
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["session_id"], session_ids[-1])

    def test_session_control_pause_resume_cancel(self):
        from simulation_backend import SimulationService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(Path(tmpdir) / "sim.db")
            session_id = service.prepare_session("wrong_way.csv", max_seconds=6)

            paused = service.control_session(session_id, "pause")
            self.assertEqual(paused["status"], "paused")

            resumed = service.control_session(session_id, "resume")
            self.assertEqual(resumed["status"], "playing")

            cancelled = service.control_session(session_id, "cancel")
            self.assertEqual(cancelled["status"], "cancelled")

            state = service.get_state(session_id)
            self.assertEqual(state["session_id"], session_id)


if __name__ == "__main__":
    unittest.main()
