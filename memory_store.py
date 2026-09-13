"""
memory_store.py — lightweight dedupe memory.

Not RAG, not a vector DB — just "have I already surfaced this URL before?"
Keyed by namespace (one per tool) so different sources don't collide.

Swap target for later: replace the JSON file with a DynamoDB table using the
same get/put shape, once this moves to AWS (Phase 5).
"""

import json
import os
from datetime import datetime, timezone

DEFAULT_MEMORY_PATH = os.environ.get("MEMORY_STORE_PATH", "memory_store.json")


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def filter_new(namespace: str, items: list[dict], path: str = DEFAULT_MEMORY_PATH) -> list[dict]:
    """
    Keep only items not seen before under this namespace, and record them as seen.

    Args:
        namespace: A short tag per source, e.g. "github_trending", "rss:aws_ml_blog".
        items: List of dicts, each expected to have a 'url' (or 'id') key that
            uniquely identifies the item.
        path: Path to the JSON memory file.

    Returns:
        The subset of items not previously recorded under this namespace.
    """
    memory = _load(path)
    seen = memory.setdefault(namespace, {})
    now = datetime.now(timezone.utc).isoformat()

    new_items = []
    for item in items:
        key = item.get("url") or item.get("id")
        if not key:
            new_items.append(item)  # can't dedupe without a key — pass through
            continue
        if key not in seen:
            seen[key] = now
            new_items.append(item)

    _save(path, memory)
    return new_items