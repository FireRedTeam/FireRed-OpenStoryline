from __future__ import annotations

from pathlib import Path


_MARKER_NAME = ".cancel_generate_ai_transition"


def _normalize_cache_root(cache_root: str | Path) -> Path:
    return Path(cache_root).expanduser()


def ai_transition_cancel_marker_path(cache_root: str | Path, session_id: str) -> Path:
    return _normalize_cache_root(cache_root) / str(session_id) / _MARKER_NAME


def set_ai_transition_cancelled(cache_root: str | Path, session_id: str) -> Path:
    marker = ai_transition_cancel_marker_path(cache_root, session_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    return marker


def clear_ai_transition_cancelled(cache_root: str | Path, session_id: str) -> None:
    marker = ai_transition_cancel_marker_path(cache_root, session_id)
    marker.unlink(missing_ok=True)


def is_ai_transition_cancelled(cache_root: str | Path, session_id: str) -> bool:
    return ai_transition_cancel_marker_path(cache_root, session_id).exists()
