from __future__ import annotations

from dataclasses import dataclass


def ensure_langgraph_runtime_compat() -> None:
    """Backfill runtime symbols expected by newer langgraph-prebuilt.

    Some dependency combinations expose only ``Runtime`` in ``langgraph.runtime``,
    while newer ``langgraph-prebuilt`` imports ``ExecutionInfo`` and ``ServerInfo``
    for type annotations. These placeholders are sufficient because the symbols are
    not used for runtime behavior in this project path.
    """

    try:
        import langgraph.runtime as runtime_mod  # type: ignore
    except Exception:
        return

    if not hasattr(runtime_mod, "ExecutionInfo"):
        @dataclass
        class ExecutionInfo:  # pragma: no cover - compatibility shim
            pass

        runtime_mod.ExecutionInfo = ExecutionInfo  # type: ignore[attr-defined]

    if not hasattr(runtime_mod, "ServerInfo"):
        @dataclass
        class ServerInfo:  # pragma: no cover - compatibility shim
            pass

        runtime_mod.ServerInfo = ServerInfo  # type: ignore[attr-defined]

    # Newer langgraph-prebuilt accesses runtime.execution_info/server_info.
    # Older langgraph Runtime does not define them, so add class-level fallbacks.
    runtime_cls = getattr(runtime_mod, "Runtime", None)
    if runtime_cls is not None:
        if not hasattr(runtime_cls, "execution_info"):
            setattr(runtime_cls, "execution_info", None)
        if not hasattr(runtime_cls, "server_info"):
            setattr(runtime_cls, "server_info", None)
