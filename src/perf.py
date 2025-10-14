from __future__ import annotations

# Simple per-challenge performance metrics, extracted to avoid circular imports
# between src.models and src.logic.

from typing import Dict

# challenge_id -> { metric_key -> total_ms }
_perf_metrics: Dict[str, Dict[str, float]] = {}


def perf_add(challenge_id: str, key: str, ms: float) -> None:
    try:
        d = _perf_metrics.setdefault(challenge_id, {})
        d[key] = d.get(key, 0.0) + float(ms)
    except Exception:
        # Never fail on metrics
        pass


def perf_get(challenge_id: str) -> dict[str, float]:
    return dict(_perf_metrics.get(challenge_id, {}))


def perf_clear(challenge_id: str) -> None:
    try:
        _perf_metrics.pop(challenge_id, None)
    except Exception:
        # Never fail on metrics
        pass
