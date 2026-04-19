from __future__ import annotations

from pathlib import Path
from typing import Any

from production_settings import ProductionSettings
from real_data import has_raw_gps_data, has_real_labeled_events, has_real_ngsim_data, has_real_workzone_data


def database_status(settings: ProductionSettings) -> dict[str, Any]:
    target = settings.database_target
    if settings.database_url:
        try:
            import psycopg  # noqa: F401

            driver = "psycopg"
            configured = True
            ready = True
            detail = "Postgres URL configured and psycopg import succeeded."
        except Exception as exc:
            driver = "missing"
            configured = True
            ready = False
            detail = f"Postgres URL configured but psycopg is unavailable: {exc}"
    else:
        driver = "sqlite3"
        configured = True
        ready = not settings.is_production_like or settings.allow_sqlite_in_production
        detail = (
            "Using SQLite fallback for local or single-node deployments."
            if ready
            else "SQLite is configured in a production-like environment without SWARMMIND_ALLOW_SQLITE_IN_PRODUCTION=true."
        )
    return {
        "kind": "database",
        "target": target,
        "driver": driver,
        "configured": configured,
        "ready": ready,
        "detail": detail,
    }


def auth_status(settings: ProductionSettings) -> dict[str, Any]:
    mode = settings.auth_mode
    if mode == "disabled":
        return {
            "kind": "auth",
            "mode": mode,
            "configured": True,
            "ready": not settings.is_production_like,
            "detail": "Authentication disabled. Safe only for localhost or private development.",
        }
    if mode == "api_key":
        return {
            "kind": "auth",
            "mode": mode,
            "configured": bool(settings.api_key),
            "ready": bool(settings.api_key),
            "detail": "Static API key protection." if settings.api_key else "Set SWARMMIND_API_KEY to enable API key auth.",
        }
    if mode == "bearer":
        ready = bool(settings.bearer_secret and settings.bearer_algorithms)
        return {
            "kind": "auth",
            "mode": mode,
            "configured": ready,
            "ready": ready,
            "detail": "Bearer auth is configured with signed token verification."
            if ready
            else "Set SWARMMIND_BEARER_SECRET and optional issuer/audience values to enable bearer auth.",
        }
    return {
        "kind": "auth",
        "mode": mode,
        "configured": False,
        "ready": False,
        "detail": f"Unsupported auth mode: {mode}",
    }


def feed_status(settings: ProductionSettings, root: Path) -> list[dict[str, Any]]:
    opencellid_dir = root / "external_data" / "opencellid"
    wzdx_dir = root / "external_data" / "wzdx"
    return [
        {
            "kind": "mapbox",
            "configured": bool(settings.mapbox_public_token),
            "ready": True,
            "detail": "Mapbox live map token is configured."
            if settings.mapbox_public_token
            else "No SWARMMIND_MAPBOX_PUBLIC_TOKEN set; UI will fall back to the local vector map.",
        },
        {
            "kind": "opencellid",
            "configured": bool(settings.opencellid_api_key) or any(opencellid_dir.glob("*.json")),
            "ready": bool(settings.opencellid_api_key) or any(opencellid_dir.glob("*.json")),
            "detail": "Tower density can use a real API key or staged exports."
            if (settings.opencellid_api_key or any(opencellid_dir.glob("*.json")))
            else "OpenCelliD is not configured; Layer 9 will remain synthetic.",
        },
        {
            "kind": "wzdx",
            "configured": bool(settings.wzdx_feed_url) or has_real_workzone_data(),
            "ready": bool(settings.wzdx_feed_url) or has_real_workzone_data(),
            "detail": "WZDx feed URL or staged GeoJSON is available."
            if (settings.wzdx_feed_url or has_real_workzone_data())
            else "No WZDx feed or staged work-zone GeoJSON found.",
        },
        {
            "kind": "ngsim",
            "configured": has_real_ngsim_data(),
            "ready": has_real_ngsim_data(),
            "detail": "Real NGSIM trajectories are staged." if has_real_ngsim_data() else "NGSIM data is missing.",
        },
        {
            "kind": "raw_gps",
            "configured": has_raw_gps_data(),
            "ready": has_raw_gps_data(),
            "detail": "Raw GPS traces are staged." if has_raw_gps_data() else "Raw GPS traces are not staged.",
        },
        {
            "kind": "labeled_events",
            "configured": has_real_labeled_events() or settings.labeled_events_manifest.exists(),
            "ready": has_real_labeled_events(),
            "detail": "Real labeled events are available."
            if has_real_labeled_events()
            else "Labeled incidents are not staged yet.",
        },
    ]


def readiness_report(settings: ProductionSettings, root: Path) -> dict[str, Any]:
    db = database_status(settings)
    auth = auth_status(settings)
    feeds = feed_status(settings, root)
    blockers = []
    for item in [db, auth, *feeds]:
        if not item["ready"]:
            blockers.append({"component": item["kind"], "detail": item["detail"]})
    return {
        "environment": settings.environment,
        "strict_startup": settings.strict_startup,
        "database": db,
        "auth": auth,
        "feeds": feeds,
        "ready": len(blockers) == 0,
        "blockers": blockers,
    }


def startup_validation_errors(settings: ProductionSettings, root: Path) -> list[str]:
    report = readiness_report(settings, root)
    errors = []
    if settings.is_production_like:
        if not report["database"]["ready"]:
            errors.append(report["database"]["detail"])
        if not report["auth"]["ready"]:
            errors.append(report["auth"]["detail"])
    return errors
