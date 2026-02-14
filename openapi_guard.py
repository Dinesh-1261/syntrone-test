# intent/openapi_guard.py
# Purpose:
# - Fetch LIVE OpenAPI spec from backend FastAPI (/openapi.json)
# - Compute a stable SHA256 hash of the spec
# - Decide whether pipeline (extract + embed) must rerun
# - Persist last seen hash under ./data/openapi_state.json

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import requests


# -----------------------------
# Paths (aligned with your repo)
# YAL/
#   data/
#   intent/
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # YAL/
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = DATA_DIR / "openapi_state.json"


# -----------------------------
# Core helpers
# -----------------------------
def fetch_openapi(openapi_url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch the LIVE OpenAPI JSON from the backend server.
    Raises requests exceptions if server is down or unreachable.
    """
    resp = requests.get(openapi_url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def compute_openapi_hash(openapi_spec: Dict[str, Any]) -> str:
    """
    Create a stable hash for the OpenAPI spec.
    Uses JSON with sorted keys to avoid hash changes due to key ordering.
    """
    raw = json.dumps(openapi_spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_state() -> Dict[str, Any]:
    """
    Load stored state (last OpenAPI hash).
    """
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        # Corrupted state -> treat as no state
        return {}


def save_state(state: Dict[str, Any]) -> None:
    """
    Persist state to disk.
    """
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def check_openapi(openapi_url: str) -> Tuple[Dict[str, Any], str, bool]:
    """
    Returns:
      spec: the fetched OpenAPI spec
      h: computed hash
      changed: True if hash != previously stored hash
    """
    spec = fetch_openapi(openapi_url=openapi_url)
    h = compute_openapi_hash(spec)

    state = load_state()
    old = state.get("openapi_hash")
    changed = (old != h)

    return spec, h, changed


def mark_built(openapi_hash: str, openapi_url: str = "") -> None:
    """
    Update state after successful extraction + embedding.
    """
    state = load_state()
    state["openapi_hash"] = openapi_hash
    if openapi_url:
        state["openapi_url"] = openapi_url
    save_state(state)
