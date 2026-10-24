import json
from pathlib import Path
from typing import Any, Dict, Optional
from app.config import DATA_DIR, CACHE_VERSION

CACHE_BASE_DIR = DATA_DIR / "cache"


def _get_cache_path(doc_id: str, kind: str) -> Path:
    return CACHE_BASE_DIR / doc_id / f"{kind}.json"


def get(doc_id: str, kind: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached data for a document and kind, or None if not cached or version mismatch."""
    cache_path = _get_cache_path(doc_id, kind)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
            # Check version
            if not isinstance(envelope, dict):
                return None
            if envelope.get("_cache_version") != CACHE_VERSION:
                return None
            return envelope.get("data")
    except Exception:
        return None


def set(doc_id: str, kind: str, data: Dict[str, Any]) -> None:
    """Store data in cache under data/cache/<doc_id>/<kind>.json with version metadata."""
    cache_path = _get_cache_path(doc_id, kind)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "_cache_version": CACHE_VERSION,
        "data": data,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)


def clear(doc_id: str) -> None:
    """Clear all cached files for a specific doc_id."""
    cache_dir = CACHE_BASE_DIR / doc_id
    if cache_dir.exists() and cache_dir.is_dir():
        for file in cache_dir.glob("*.json"):
            try:
                file.unlink()
            except OSError:
                pass
