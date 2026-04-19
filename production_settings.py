from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ProductionSettings:
    environment: str
    database_url: str | None
    sqlite_path: Path
    auth_mode: str
    api_key: str | None
    bearer_issuer: str | None
    bearer_audience: str | None
    bearer_secret: str | None
    bearer_algorithms: tuple[str, ...]
    allow_insecure_localhost: bool
    strict_startup: bool
    allow_sqlite_in_production: bool
    mapbox_public_token: str | None
    mapbox_style: str
    opencellid_api_key: str | None
    wzdx_feed_url: str | None
    labeled_events_manifest: Path

    @property
    def database_target(self) -> str:
        if self.database_url:
            return self.database_url
        return str(self.sqlite_path)

    @property
    def is_production_like(self) -> bool:
        return self.environment.lower() in {"prod", "production", "staging"}


def load_settings() -> ProductionSettings:
    sqlite_path = Path(os.environ.get("SWARMMIND_SIM_DB", str(ROOT / "simulation_state.db")))
    labeled_manifest = Path(
        os.environ.get(
            "SWARMMIND_LABELED_EVENTS_MANIFEST",
            str(ROOT / "external_data" / "labeled_events" / "manifest.json"),
        )
    )
    return ProductionSettings(
        environment=os.environ.get("SWARMMIND_ENV", "development"),
        database_url=os.environ.get("SWARMMIND_DATABASE_URL"),
        sqlite_path=sqlite_path,
        auth_mode=os.environ.get("SWARMMIND_AUTH_MODE", "disabled").strip().lower(),
        api_key=os.environ.get("SWARMMIND_API_KEY"),
        bearer_issuer=os.environ.get("SWARMMIND_BEARER_ISSUER"),
        bearer_audience=os.environ.get("SWARMMIND_BEARER_AUDIENCE"),
        bearer_secret=os.environ.get("SWARMMIND_BEARER_SECRET"),
        bearer_algorithms=tuple(
            part.strip()
            for part in os.environ.get("SWARMMIND_BEARER_ALGORITHMS", "HS256").split(",")
            if part.strip()
        ),
        allow_insecure_localhost=_env_bool("SWARMMIND_ALLOW_INSECURE_LOCALHOST", True),
        strict_startup=_env_bool("SWARMMIND_STRICT_STARTUP", False),
        allow_sqlite_in_production=_env_bool("SWARMMIND_ALLOW_SQLITE_IN_PRODUCTION", False),
        mapbox_public_token=os.environ.get("SWARMMIND_MAPBOX_PUBLIC_TOKEN"),
        mapbox_style=os.environ.get("SWARMMIND_MAPBOX_STYLE", "mapbox://styles/mapbox/dark-v11"),
        opencellid_api_key=os.environ.get("OPENCELLID_API_KEY"),
        wzdx_feed_url=os.environ.get("SWARMMIND_WZDX_FEED_URL"),
        labeled_events_manifest=labeled_manifest,
    )
