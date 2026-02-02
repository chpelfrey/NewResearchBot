"""Research log - JSON log of questions, responses, timestamps, and response times."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# Default log path: project root or cwd
DEFAULT_LOG_PATH = os.environ.get("RESEARCH_LOG_PATH", "research_log.json")


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, remove punctuation for comparison."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _word_set(s: str) -> set[str]:
    return set(_normalize(s).split()) - {"a", "an", "the", "is", "are", "what", "how", "when", "where", "why", "who"}


def _relevance_score(query: str, logged_query: str) -> float:
    """
    Score 0-1: 1 = exact match, high = strong word overlap or containment.
    """
    nq = _normalize(query)
    nlogged = _normalize(logged_query)
    if not nq:
        return 0.0
    if nq == nlogged:
        return 1.0
    if nq in nlogged or nlogged in nq:
        return 0.9
    q_words = _word_set(query)
    log_words = _word_set(logged_query)
    if not q_words:
        return 0.0
    overlap = len(q_words & log_words) / len(q_words)
    return min(overlap * 1.2, 1.0)  # Slight boost for partial overlap


def get_log_path() -> Path:
    """Return path to the research log file."""
    return Path(DEFAULT_LOG_PATH).resolve()


def load_entries() -> list[dict]:
    """Load all log entries from the JSON file. Returns [] if missing or invalid."""
    path = get_log_path()
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_entry(
    query: str,
    response: str,
    response_time_seconds: float,
    timestamp: str | None = None,
) -> None:
    """
    Append one Q&A entry to the research log.
    
    Args:
        query: User question.
        response: Bot answer.
        response_time_seconds: Time taken to produce the response.
        timestamp: ISO timestamp (default: now UTC).
    """
    path = get_log_path()
    entries = load_entries()
    entries.append({
        "query": query,
        "response": response,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "response_time_seconds": round(response_time_seconds, 2),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def get_relevant_entries(
    query: str,
    *,
    min_score: float = 0.4,
    max_entries: int = 5,
) -> list[dict]:
    """
    Return past log entries relevant to the given query, sorted by relevance (newest first for ties).
    
    Args:
        query: Current user question.
        min_score: Minimum relevance score (0-1).
        max_entries: Maximum number of entries to return.
    
    Returns:
        List of dicts with keys: query, response, timestamp, response_time_seconds, relevance_score
    """
    entries = load_entries()
    if not query or not entries:
        return []
    scored = []
    for e in entries:
        logged_q = e.get("query") or ""
        score = _relevance_score(query, logged_q)
        if score >= min_score:
            scored.append({
                **e,
                "relevance_score": round(score, 2),
            })
    scored.sort(key=lambda x: (-x["relevance_score"], x.get("timestamp", "")), reverse=False)
    return scored[:max_entries]
