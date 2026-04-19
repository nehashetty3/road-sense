from __future__ import annotations

import base64
import hmac
import json
import time
from typing import Iterable

from fastapi import HTTPException, Request

from production_settings import ProductionSettings


def _is_local_request(request: Request) -> bool:
    client = request.client.host if request.client else ""
    return client in {"127.0.0.1", "::1", "localhost", ""}


def _public_paths() -> Iterable[str]:
    return {
        "/",
        "/health",
        "/readiness",
        "/dependencies",
        "/docs",
        "/openapi.json",
    }


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _verify_hs256_token(token: str, settings: ProductionSettings) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Malformed bearer token.")
    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid bearer token encoding: {exc}") from exc

    alg = header.get("alg")
    if alg not in settings.bearer_algorithms:
        raise HTTPException(status_code=401, detail=f"Unsupported bearer algorithm: {alg}")
    if alg != "HS256":
        raise HTTPException(status_code=401, detail="Only HS256 bearer tokens are currently supported.")

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(settings.bearer_secret.encode("utf-8"), signing_input, "sha256").digest()
    actual_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise HTTPException(status_code=401, detail="Invalid bearer token signature.")

    exp = payload.get("exp")
    if exp is None or float(exp) <= time.time():
        raise HTTPException(status_code=401, detail="Bearer token is expired or missing exp.")
    if settings.bearer_issuer and payload.get("iss") != settings.bearer_issuer:
        raise HTTPException(status_code=401, detail="Bearer token issuer mismatch.")
    aud = payload.get("aud")
    if settings.bearer_audience:
        if isinstance(aud, list):
            valid_aud = settings.bearer_audience in aud
        else:
            valid_aud = aud == settings.bearer_audience
        if not valid_aud:
            raise HTTPException(status_code=401, detail="Bearer token audience mismatch.")
    return payload


def enforce_request_auth(request: Request, settings: ProductionSettings) -> None:
    path = request.url.path
    if path in _public_paths() or path.startswith("/static/"):
        return
    if settings.allow_insecure_localhost and _is_local_request(request):
        return

    mode = settings.auth_mode
    if mode == "disabled":
        return

    if mode == "api_key":
        provided = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not settings.api_key or not provided or not hmac.compare_digest(provided, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key.")
        return

    if mode == "bearer":
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token.")
        token = auth_header[7:].strip()
        if not settings.bearer_secret:
            raise HTTPException(status_code=503, detail="Bearer auth is not fully configured.")
        _verify_hs256_token(token, settings)
        return

    raise HTTPException(status_code=503, detail=f"Unsupported auth mode: {mode}")
