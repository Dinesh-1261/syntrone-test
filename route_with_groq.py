# intent/route_with_groq.py
"""
route_with_groq.py (UPDATED v6) — Method-specific Chroma + Module Registry + Groq Re-ranker

Flow:
1) Groq classify: {method, primary_module, related_module} (module canonicalized via registry)
2) Chroma semantic search on method-specific collection (+ optional module filter)
3) Groq re-rank Top-K candidates and select the SINGLE best endpoint (by operation_id)

Dependencies:
  pip install groq python-dotenv chromadb sentence-transformers
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config (folder-structure aligned)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = str(DATA_DIR / "chroma_db")
REGISTRY_PATH = DATA_DIR / "module_registry.json"

COLLECTIONS_BY_METHOD = {
    "GET": "fastapi_endpoints_get",
    "POST": "fastapi_endpoints_post",
    "PUT": "fastapi_endpoints_put",
    "DELETE": "fastapi_endpoints_delete",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE"]


# -----------------------------
# Registry helpers
# -----------------------------
def load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"canonical_modules": [], "aliases": {}}


# -----------------------------
# Helpers
# -----------------------------
def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def normalize_method(m: str) -> str:
    m = (m or "").strip().upper()
    if m == "PATCH":
        return "PUT"
    if m not in {"GET", "POST", "PUT", "DELETE"}:
        return "GET"
    return m


def _normalize_module(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def canonicalize_module(raw: str, registry: Dict[str, Any]) -> str:
    """
    Map raw module (from Groq) into canonical module key using registry aliases.
    If cannot map reliably, return "" so we don't filter and accidentally wipe results.
    """
    raw_key = _normalize_module(raw)
    if not raw_key:
        return ""

    aliases = registry.get("aliases") or {}
    canonicals = set(registry.get("canonical_modules") or [])

    if raw_key in aliases:
        return aliases[raw_key]
    if raw_key in canonicals:
        return raw_key

    # Heuristic: containment match (receipt -> direct_receipts)
    for c in canonicals:
        if raw_key and raw_key in c:
            return c

    return ""


def _safe_json_extract(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON in LLM response: {text[:200]}")
    return json.loads(m.group(0))


def _module_match_score(module: str, md: Dict[str, Any]) -> int:
    module = _normalize_module(module)
    if not module:
        return 0

    md_module = _normalize_module(md.get("module", ""))
    if md_module and md_module == module:
        return 3

    path = (md.get("path") or "").lower()
    summary = (md.get("summary") or "").lower()
    tags = (md.get("tags") or "").lower()

    if module in path:
        return 2
    if module in summary or module in tags:
        return 1
    return 0


def _collection_has_module_field(collection) -> bool:
    try:
        peek = collection.peek(1)
        mds = peek.get("metadatas") or []
        if not mds:
            return False
        return "module" in (mds[0] or {})
    except Exception:
        return False


def _get_collection_for_method(client: chromadb.PersistentClient, method: str):
    cname = COLLECTIONS_BY_METHOD.get(method)
    if not cname:
        raise RuntimeError(f"No collection mapped for method: {method}")
    return client.get_collection(name=cname)


def _is_empty_query_result(res: Dict[str, Any]) -> bool:
    docs = res.get("documents", [[]])
    return (not docs) or (not docs[0])


# -----------------------------
# Groq LLM classification
# -----------------------------
def classify_with_groq(user_query: str, registry: Dict[str, Any]) -> Dict[str, str]:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment/.env")

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = Groq(api_key=api_key)

    known_modules = registry.get("canonical_modules") or []

    system_prompt = (
        "You are an API routing classifier.\n"
        "Extract the intended HTTP method and PRIMARY business module from the user's request.\n"
        "If the request is like 'create X under Y', then primary_module=X and related_module=Y.\n"
        "Return ONLY valid JSON. No markdown. No explanations.\n\n"
        f"Allowed methods: {ALLOWED_METHODS}\n"
        "IMPORTANT: PATCH is NOT supported. For partial updates, output PUT.\n"
        f"Known modules (canonical keys): {known_modules}\n\n"
        "Method guidance:\n"
        "- GET: list/show/fetch/view/get\n"
        "- POST: create/add/insert\n"
        "- PUT: update/replace/change\n"
        "- DELETE: delete/remove\n\n"
        "Output schema:\n"
        '{ "method": "...", "primary_module": "...", "related_module": "" }'
    )

    user_prompt = f'User query: "{user_query}"'

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content
    data = _safe_json_extract(text)

    method = normalize_method(str(data.get("method", "")))
    primary_raw = str(data.get("primary_module", ""))
    related_raw = str(data.get("related_module", ""))

    primary = canonicalize_module(primary_raw, registry)
    related = canonicalize_module(related_raw, registry)

    return {"method": method, "primary_module": primary, "related_module": related}


# -----------------------------
# Chroma Search
# -----------------------------
def _chroma_query(collection, qvec, n_results, where_filter: Optional[Dict[str, Any]] = None):
    """
    Chroma strict validator does NOT accept where={}
    So we include 'where' only when it's non-empty.
    """
    kwargs = dict(
        query_embeddings=[qvec],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter
    return collection.query(**kwargs)


def search_endpoints(
    user_query: str,
    method: str,
    primary_module: str,
    top_k: int = 5,
    embedder: Optional[SentenceTransformer] = None,
) -> List[Dict[str, Any]]:
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)

    qvec = embedder.encode([user_query], normalize_embeddings=True)[0].tolist()

    client = _chroma_client()
    collection = _get_collection_for_method(client, method)

    has_module = _collection_has_module_field(collection)
    wide_k = max(top_k * 6, 30)

    where_filter = None
    if has_module and primary_module:
        where_filter = {"module": {"$eq": primary_module}}

    # Query with filter (if any)
    res = _chroma_query(collection, qvec, wide_k, where_filter=where_filter)

    # If empty -> retry WITHOUT filter
    if _is_empty_query_result(res):
        res = _chroma_query(collection, qvec, wide_k, where_filter=None)

    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    documents = res.get("documents", [[]])[0]

    items: List[Tuple[int, float, Dict[str, Any]]] = []
    for md, dist, doc in zip(metadatas, distances, documents):
        md = dict(md or {})
        md["embedding_text"] = doc
        score = _module_match_score(primary_module, md) if primary_module else 0
        items.append((score, float(dist), md))

    items.sort(key=lambda x: (-x[0], x[1]))

    out: List[Dict[str, Any]] = []
    for score, dist, md in items[:top_k]:
        out.append(
            {
                "module_match": score,
                "distance": dist,
                "method": md.get("method", method),
                "path": md.get("path"),
                "summary": md.get("summary"),
                "tags": md.get("tags"),
                "operation_id": md.get("operation_id"),
                "module": md.get("module", ""),
            }
        )
    return out


# -----------------------------
# Groq Re-ranker (Top-K -> single best)
# -----------------------------
def rerank_with_groq(user_query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask Groq to choose exactly ONE best endpoint from Top-K.
    Returns the chosen candidate dict (adds _llm_reason).
    """
    if not candidates:
        return {}

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment/.env")

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = Groq(api_key=api_key)

    packed = []
    for c in candidates:
        packed.append(
            {
                "operation_id": c.get("operation_id", ""),
                "method": c.get("method", ""),
                "path": c.get("path", ""),
                "summary": c.get("summary", ""),
                "module": c.get("module", ""),
                "tags": c.get("tags", ""),
            }
        )

    system_prompt = (
        "You are an API endpoint selector.\n"
        "Given a user query and candidate endpoints, choose the SINGLE best endpoint.\n"
        "Pick the endpoint that best matches the user's intent.\n"
        "Return ONLY JSON.\n\n"
        'Output schema: {"operation_id": "...", "reason": "short"}\n'
        "Rules (STRICT):\n"
        "- operation_id MUST be exactly one of the candidates' operation_id values.\n"
        "- If the user query mentions a type, category, status, or qualifier "
        "(e.g., income, expense, vendor, active, client type), "
        "PREFER endpoints that accept path or query parameters over generic endpoints.\n"
        "- If multiple endpoints are close, prefer the most direct action "
        "(update > revise > revert > add > remove).\n"
        "- Do NOT choose generic list or export endpoints when a more specific filtered endpoint exists.\n"
    )

    user_payload = {"user_query": user_query, "candidates": packed}
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    data = _safe_json_extract(resp.choices[0].message.content)
    chosen_op = str(data.get("operation_id", "")).strip()
    reason = str(data.get("reason", "")).strip()

    # Select by operation_id (stable even if ordering changes)
    for c in candidates:
        if (c.get("operation_id") or "") == chosen_op:
            out = dict(c)
            out["_llm_reason"] = reason
            return out

    # Fallback: if Groq returns something unexpected, pick top-1 candidate
    out = dict(candidates[0])
    out["_llm_reason"] = f"(fallback top-1) {reason}".strip()
    return out


# -----------------------------
# Output (CLI helper)
# -----------------------------
def pretty_print(user_query: str, classified: Dict[str, str], results: List[Dict[str, Any]]):
    print("\n" + "=" * 95)
    print(f"User Query : {user_query}")
    print(
        "LLM Output : "
        f"method={classified['method']} | primary_module={classified['primary_module']} | related_module={classified['related_module']}"
    )
    print(f"Chroma Collection : {COLLECTIONS_BY_METHOD.get(classified['method'])}")
    print("=" * 95)

    if not results:
        print("No candidates found.")
        return

    print("\nTop candidates (lower distance = better; module_match higher = better):\n")
    for i, r in enumerate(results, start=1):
        mm = r.get("module_match", 0)
        print(f"{i}. module_match={mm} | distance={r['distance']:.4f} | {r['method']} {r['path']}")
        if r.get("summary"):
            print(f"   summary: {r['summary']}")
        if r.get("tags"):
            print(f"   tags: {r['tags']}")
        if r.get("module"):
            print(f"   module: {r['module']}")
        if r.get("operation_id"):
            print(f"   operation_id: {r['operation_id']}")
        print()


# -----------------------------
# Main (CLI mode)
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python intent/route_with_groq.py "your query" [top_k]')
        sys.exit(1)

    user_query = sys.argv[1]
    top_k = 5
    if len(sys.argv) >= 3:
        try:
            top_k = int(sys.argv[2])
        except ValueError:
            top_k = 5

    registry = load_registry()
    embedder = SentenceTransformer(EMBED_MODEL)

    classified = classify_with_groq(user_query, registry)

    results = search_endpoints(
        user_query=user_query,
        method=classified["method"],
        primary_module=classified["primary_module"],
        top_k=top_k,
        embedder=embedder,
    )

    pretty_print(user_query, classified, results)

    if results:
        chosen = rerank_with_groq(user_query, results)
        if chosen:
            print("\nFINAL (LLM-selected):")
            print(f"- {chosen.get('method')} {chosen.get('path')}")
            print(f"- operation_id: {chosen.get('operation_id')}")
            if chosen.get("_llm_reason"):
                print(f"- reason: {chosen.get('_llm_reason')}")


if __name__ == "__main__":
    main()
