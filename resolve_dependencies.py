from typing import Any, Dict, List
from intent.post_dependency_map import POST_DEPENDENCY_MAP


def get_dependencies_for_endpoint(endpoint: str, method: str) -> List[Dict[str, Any]]:
    m = (method or "").upper().strip()

    # Only attach dependencies for write methods
    if m not in {"POST", "PUT"}:
        return []

    entry = POST_DEPENDENCY_MAP.get(endpoint)
    if not entry:
        return []

    return entry.get("depends_on", [])
