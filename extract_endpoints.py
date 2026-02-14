# intent/extract_endpoints.py
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

# -----------------------------
# Config
# -----------------------------
OPENAPI_URL = "http://127.0.0.1:8000/openapi.json"

# Folder-structure aligned outputs:
# langgraph_app/
#   data/
#     endpoint_catalog.json
#     module_registry.json
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = DATA_DIR / "endpoint_catalog.json"
REGISTRY_PATH = DATA_DIR / "module_registry.json"

JUNK_PATH_SEGMENTS = {"api", "meta", "test", "v1", "v2"}


# -----------------------------
# Normalization primitives (generic; not business hard-coded)
# -----------------------------
def _clean_text(s: str) -> str:
    s = (s or "").strip().lower()
    # keep letters, numbers, separators, spaces
    s = re.sub(r"[^a-z0-9_\-\s/]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_canonical_key(s: str) -> str:
    """
    Canonical key format: lowercase, underscore-separated.
    No business mapping here — purely shape normalization.
    """
    s = _clean_text(s)
    s = s.replace("/", " ")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s


def _singular_plural_variants(key: str) -> Set[str]:
    """
    Lightweight pluralization support.
    Not perfect English NLP — just practical for API module keys.
    """
    v = set()
    k = key.strip("_")
    if not k:
        return v

    v.add(k)

    # if endswith s, add singular (drop trailing s)
    if k.endswith("s") and len(k) > 2:
        v.add(k[:-1])

    # if not endswith s, add plural (+s)
    if not k.endswith("s"):
        v.add(k + "s")

    return v


def generate_aliases_for_canonical(canonical: str) -> Set[str]:
    """
    For a canonical module key like 'direct_receipts', generate common aliases:
      direct_receipts
      direct-receipts
      direct receipts
      directreceipts
      direct_receipt (singular)
      direct-receipt
      direct receipt
      directreceipt
    This avoids hard-coded alias maps while preserving stability.
    """
    aliases: Set[str] = set()

    # Add singular/plural variants of canonical itself
    base_variants = _singular_plural_variants(canonical)

    for base in base_variants:
        # underscore, hyphen, space, concatenated
        aliases.add(base)
        aliases.add(base.replace("_", "-"))
        aliases.add(base.replace("_", " "))
        aliases.add(base.replace("_", ""))

        # also accept already-normalized forms of these
        aliases.add(_to_canonical_key(base.replace("_", "-")))
        aliases.add(_to_canonical_key(base.replace("_", " ")))
        aliases.add(_to_canonical_key(base.replace("_", "")))

    # Add cleaned forms
    aliases = {a for a in aliases if a and a not in {"api", "meta", "test", "v1", "v2"}}
    return aliases


# -----------------------------
# Registry
# -----------------------------
def load_registry() -> Dict[str, Any]:
    """
    Registry schema:
      {
        "canonical_modules": [...],
        "aliases": { "alias_key": "canonical_key", ... },
        "last_updated": "YYYY-MM-DD"
      }
    """
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {"canonical_modules": [], "aliases": {}, "last_updated": ""}


def save_registry(reg: Dict[str, Any]) -> None:
    reg["canonical_modules"] = sorted(set(reg.get("canonical_modules") or []))
    reg["aliases"] = reg.get("aliases") or {}
    reg["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_to_canonical(raw: str, reg: Dict[str, Any]) -> str:
    """
    Convert a raw module candidate to canonical using registry aliases.
    If unknown, return canonical-shaped key (not mapped).
    """
    raw_key = _to_canonical_key(raw)
    if not raw_key:
        return ""

    aliases = reg.get("aliases") or {}
    # direct alias hit
    if raw_key in aliases:
        return aliases[raw_key]

    # if raw itself is already canonical in registry
    canonicals = set(reg.get("canonical_modules") or [])
    if raw_key in canonicals:
        return raw_key

    return raw_key


def update_registry_with_candidates(candidates: Set[str], reg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds new canonical modules and auto-generates aliases for them.
    Does NOT remove existing entries (safe, append-only behavior).
    """
    canonicals = set(reg.get("canonical_modules") or [])
    aliases: Dict[str, str] = dict(reg.get("aliases") or {})

    # Add new canonicals
    for c in sorted(candidates):
        c_key = _to_canonical_key(c)
        if not c_key:
            continue
        if c_key in {"api", "meta", "test", "v1", "v2"}:
            continue
        canonicals.add(c_key)

    # Auto-generate aliases for all canonicals (including newly added)
    for c in sorted(canonicals):
        for a in generate_aliases_for_canonical(c):
            a_key = _to_canonical_key(a)
            if not a_key:
                continue
            # Do not overwrite existing alias mapping if present
            aliases.setdefault(a_key, c)

    reg["canonical_modules"] = sorted(canonicals)
    reg["aliases"] = aliases
    return reg


# -----------------------------
# Module inference
# -----------------------------
def _module_from_tags(tags: Any) -> str:
    if not tags or not isinstance(tags, list):
        return ""
    if not tags:
        return ""
    return str(tags[0])


def _path_segments(path: str) -> List[str]:
    if not path:
        return []
    return [p for p in path.strip("/").split("/") if p]


def _candidate_modules_from_path(path: str) -> List[str]:
    """
    Produces multiple candidates from path:
    - first non-junk segment (umbrella like 'finance')
    - also scan early segments for more specific nouns (like 'direct-receipts')
    We return multiple candidates; later we pick best.
    """
    parts = _path_segments(path)
    cleaned = [p for p in parts if p.lower() not in JUNK_PATH_SEGMENTS]
    if not cleaned:
        return []

    candidates: List[str] = []

    # Always include first segment (common module grouping)
    candidates.append(cleaned[0])

    # Also include first few segments if they look meaningful
    for seg in cleaned[:4]:
        # skip path params
        if seg.startswith("{") and seg.endswith("}"):
            continue
        candidates.append(seg)

    return candidates


def infer_module(path: str, tags: Any, summary: str, operation_id: str, reg: Dict[str, Any]) -> str:
    """
    Priority:
    1) tags (best semantic grouping)
    2) path candidates (try more specific ones first)
    3) summary/operation_id fallback (weak signal)
    All resolved via registry aliases → canonical.
    """
    # 1) tags
    tag_candidate = _module_from_tags(tags)
    if tag_candidate:
        return resolve_to_canonical(tag_candidate, reg)

    # 2) path candidates (prefer specific candidates)
    path_candidates = _candidate_modules_from_path(path)
    # Prefer longer/more specific candidates first (e.g., 'direct-receipts' over 'finance')
    path_candidates = sorted(set(path_candidates), key=lambda x: (-len(x), x))

    for cand in path_candidates:
        canon = resolve_to_canonical(cand, reg)
        if canon:
            return canon

    # 3) text fallback: attempt match against known canonicals
    blob = _clean_text(f"{summary or ''} {operation_id or ''}")
    canonicals = reg.get("canonical_modules") or []
    for c in canonicals:
        if not c:
            continue
        if c in blob or c.replace("_", " ") in blob:
            return c

    return ""


# -----------------------------
# Main
# -----------------------------
def main():
    spec = requests.get(OPENAPI_URL, timeout=30).json()
    paths = spec.get("paths", {}) or {}

    # Load existing registry (if any)
    reg = load_registry()

    # First pass: discover module candidates from tags + path segments
    discovered: Set[str] = set()
    for path, methods in paths.items():
        for _, meta in (methods or {}).items():
            meta = meta or {}
            tags = meta.get("tags") or []
            if tags:
                discovered.add(str(tags[0]))
        # add path-derived candidates too
        for cand in _candidate_modules_from_path(path):
            discovered.add(cand)

    # Update registry with discovered candidates (auto-alias generation)
    reg = update_registry_with_candidates(discovered, reg)
    save_registry(reg)

    # Build endpoint catalog with module resolved using registry
    catalog: List[Dict[str, Any]] = []

    for path, methods in paths.items():
        for method, meta in (methods or {}).items():
            m = (method or "").upper()
            if m not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue

            meta = meta or {}
            operation_id = meta.get("operationId") or ""
            summary = meta.get("summary") or ""
            description = meta.get("description") or ""
            tags = meta.get("tags") or []

            module = infer_module(
                path=path,
                tags=tags,
                summary=summary,
                operation_id=operation_id,
                reg=reg,
            )

            catalog.append(
                {
                    "method": m,
                    "path": path,
                    "operation_id": operation_id,
                    "summary": summary,
                    "description": description,
                    "tags": tags,
                    "parameters": meta.get("parameters") or [],
                    "request_body": meta.get("requestBody") or None,
                    "module": module,
                }
            )

    OUT_JSON.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Extracted {len(catalog)} endpoints")
    print(f"📄 Saved to: {OUT_JSON}")
    print(f"📘 Registry updated: {REGISTRY_PATH} (modules={len(reg.get('canonical_modules') or [])})")

    # Quick preview
    for i, ep in enumerate(catalog[:10], start=1):
        print(f"{i:02d}. {ep['method']} {ep['path']} | module={ep.get('module','')} | {ep.get('summary','')}")

    missing_mod = sum(1 for ep in catalog if not ep.get("module"))
    if missing_mod:
        print(f"⚠️ Note: {missing_mod} endpoints have empty module (tags/path inference not found).")


if __name__ == "__main__":
    main()
