# intent/embed_to_chroma.py
# (UPDATED v4) — method-specific collections + NEW Chroma PersistentClient
# Folder-structure aligned: reads/writes under ./data/

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config (folder-structure aligned)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = DATA_DIR / "endpoint_catalog.json"

# Keep DB inside project under data/
CHROMA_DIR = str(DATA_DIR / "chroma_db")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Create 4 separate collections (method "silos")
COLLECTIONS_BY_METHOD = {
    "GET": "fastapi_endpoints_get",
    "POST": "fastapi_endpoints_post",
    "PUT": "fastapi_endpoints_put",
    "DELETE": "fastapi_endpoints_delete",
}

ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE"}


# -----------------------------
# Helpers
# -----------------------------
def normalize_method(method: str) -> str:
    m = (method or "").strip().upper()
    if m == "PATCH":
        return "PUT"
    if m not in ALLOWED_METHODS:
        return ""  # unsupported
    return m


def build_text_for_embedding(ep: Dict[str, Any]) -> str:
    """Create a compact, semantic text representation of an endpoint."""
    method = ep.get("method", "")
    path = ep.get("path", "")
    summary = ep.get("summary") or ""
    desc = ep.get("description") or ""
    tags = ", ".join(ep.get("tags") or [])
    module = ep.get("module") or ""

    parts = [
        f"method: {method}",
        f"path: {path}",
        f"summary: {summary}",
    ]
    if module:
        parts.append(f"module: {module}")
    if tags:
        parts.append(f"tags: {tags}")
    if desc:
        parts.append(f"description: {desc}")

    return " | ".join(parts)


def stable_id(ep: Dict[str, Any]) -> str:
    """
    Stable UUID per endpoint.
    Uses method + path + operation_id so re-indexing does not create duplicates.
    """
    op_id = ep.get("operation_id") or ""
    key = f"{ep.get('method','')}::{ep.get('path','')}::{op_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def chroma_client() -> chromadb.PersistentClient:
    # NEW Chroma client (no legacy Settings)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def main():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing Step 1 output: {CATALOG_PATH}")

    catalog: List[Dict[str, Any]] = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(catalog)} endpoints from {CATALOG_PATH}")

    # Normalize methods upfront (PATCH->PUT, ignore unsupported)
    normalized_catalog: List[Dict[str, Any]] = []
    dropped = 0
    for ep in catalog:
        m = normalize_method(ep.get("method", ""))
        if not m:
            dropped += 1
            continue
        ep2 = dict(ep)
        ep2["method"] = m
        normalized_catalog.append(ep2)

    if dropped:
        print(f"⚠️ Dropped {dropped} endpoints due to unsupported methods")

    # Embedding model
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Loaded model: {MODEL_NAME} (dim={dim})")

    # Chroma client
    client = chroma_client()

    # Recreate collections cleanly (safe for dev)
    for _, cname in COLLECTIONS_BY_METHOD.items():
        try:
            client.delete_collection(cname)
            print(f"Deleted existing collection: {cname}")
        except Exception:
            pass

    collections = {m: client.create_collection(name=cname) for m, cname in COLLECTIONS_BY_METHOD.items()}
    print(f"✅ Chroma collections ready at {CHROMA_DIR}")
    for m, cname in COLLECTIONS_BY_METHOD.items():
        print(f"   - {m}: {cname}")

    # Prepare all texts + embeddings once (efficient)
    texts = [build_text_for_embedding(ep) for ep in normalized_catalog]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Group inserts by method (each group goes to its own collection)
    grouped: Dict[str, List[Tuple[str, str, Dict[str, Any], List[float]]]] = {
        "GET": [],
        "POST": [],
        "PUT": [],
        "DELETE": [],
    }

    for ep, text, emb in zip(normalized_catalog, texts, embeddings):
        method = ep.get("method")
        if method not in grouped:
            continue

        md = {
            "method": ep.get("method"),
            "path": ep.get("path"),
            "operation_id": ep.get("operation_id") or "",
            "summary": ep.get("summary") or "",
            "description": ep.get("description") or "",
            "tags": ", ".join(ep.get("tags") or []),
            # add module for downstream filtering/scoring
            "module": ep.get("module") or "",
        }

        grouped[method].append((stable_id(ep), text, md, emb.tolist()))

    # Insert per collection
    total = 0
    for method, rows in grouped.items():
        if not rows:
            print(f"ℹ️ No endpoints for method={method}, skipping insert.")
            continue

        ids = [r[0] for r in rows]
        documents = [r[1] for r in rows]
        metadatas = [r[2] for r in rows]
        embs = [r[3] for r in rows]

        collections[method].add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embs,
        )

        total += len(ids)
        print(f"✅ Stored {len(ids)} endpoints in collection '{COLLECTIONS_BY_METHOD[method]}'")

    print(f"\n✅ Done. Stored total {total} endpoints across 4 collections.")
    print(f"✅ Persisted vectors to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
