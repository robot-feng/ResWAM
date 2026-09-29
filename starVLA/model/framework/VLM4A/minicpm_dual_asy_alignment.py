# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Control-step alignment schedules for MiniCPM DualAsy.

The functions in this module are pure so training-sample alignment and
inference publication can use the same source-step rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ALIGNMENT_MODES = ("synchronous", "fixed_step_delay", "trace_replay", "wall_clock")


def _normalize_activation_steps(events: dict[int, int]) -> dict[int, int]:
    normalized = {int(source): int(activation) for source, activation in events.items()}
    if normalized.get(0) != 0:
        raise ValueError("trace_replay must include the synchronous bootstrap event 0 -> 0")
    previous_activation = -1
    for source, activation in sorted(normalized.items()):
        if source < 0 or activation < source:
            raise ValueError(
                "trace source and activation steps must be nonnegative and activation >= source"
            )
        if activation < previous_activation:
            raise ValueError(
                "trace activation steps must be monotonic with their source steps"
            )
        previous_activation = activation
    return normalized


def fixed_activation_steps(
    refresh_interval: int,
    latency_steps: int,
    max_control_step: int,
    execution_horizon: int = 1,
) -> dict[int, int]:
    """Return ``source_step -> activation_step`` for a serialized worker.

    The first refresh at step zero is an awaited bootstrap. Subsequent jobs are
    requested every ``refresh_interval`` control steps and require
    ``latency_steps`` control steps of worker service. If service takes longer
    than the interval, requests queue behind the prior job. A completed result
    becomes action-visible at the next policy-call boundary, determined by
    ``execution_horizon``.
    """
    refresh_interval = int(refresh_interval)
    latency_steps = int(latency_steps)
    max_control_step = int(max_control_step)
    execution_horizon = int(execution_horizon)
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if latency_steps < 0:
        raise ValueError("latency_steps must be >= 0")
    if max_control_step < 0:
        raise ValueError("max_control_step must be >= 0")
    if execution_horizon < 1:
        raise ValueError("execution_horizon must be >= 1")

    events = {0: 0}
    previous_completion = 0
    source_step = refresh_interval
    while source_step <= max_control_step:
        start_step = max(source_step, previous_completion)
        ready_step = start_step + latency_steps
        activation_step = (
            (ready_step + execution_horizon - 1) // execution_horizon
        ) * execution_horizon
        events[source_step] = activation_step
        previous_completion = ready_step
        source_step += refresh_interval
    return events


def source_step_at(
    control_step: int,
    *,
    mode: str,
    refresh_interval: int,
    fixed_latency_steps: int | None = None,
    trace_activation_steps: dict[int, int] | None = None,
    execution_horizon: int = 1,
) -> int:
    """Return the newest VLM source step available to an action at ``t``.

    Offline data construction cannot predict a future wall-clock completion.
    For that mode callers must first calibrate a trace and select
    ``trace_replay`` explicitly.
    """
    control_step = int(control_step)
    refresh_interval = int(refresh_interval)
    if control_step < 0:
        raise ValueError("control_step must be >= 0")
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"alignment mode must be one of {ALIGNMENT_MODES}, got {mode!r}")

    if mode == "wall_clock":
        raise ValueError(
            "offline anchor construction cannot predict wall-clock readiness; "
            "use trace_replay or a controlled delay"
        )
    if mode == "trace_replay":
        if not trace_activation_steps:
            raise ValueError("trace_replay requires source_step -> activation_step events")
        events = _normalize_activation_steps(trace_activation_steps)
    else:
        latency = 0 if mode == "synchronous" else fixed_latency_steps
        if latency is None:
            raise ValueError("fixed_step_delay requires fixed_latency_steps")
        events = fixed_activation_steps(
            refresh_interval,
            int(latency),
            control_step,
            execution_horizon,
        )

    eligible = [source for source, activation in events.items() if activation <= control_step]
    if not eligible:
        raise RuntimeError(f"no VLM source is active at control_step={control_step}")
    return max(eligible)


def _find_step_trace(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        trace = value.get("async_step_trace")
        if trace and isinstance(trace, list) and all(isinstance(row, dict) for row in trace):
            return trace
        events = value.get("activation_events")
        if events and isinstance(events, list) and all(isinstance(row, dict) for row in events):
            return events
        for child in value.values():
            found = _find_step_trace(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_step_trace(child)
            if found is not None:
                return found
    return None


def load_trace_activation_steps(path: str | Path) -> dict[int, int]:
    """Load explicit events or infer activation transitions from pilot JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"alignment trace does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _find_step_trace(payload)
    if rows is None:
        raise ValueError(f"no async_step_trace or activation_events found in {path}")

    activation_steps: dict[int, int] = {}
    previous_source = None
    for row in rows:
        if "source_step" in row and "activation_step" in row:
            source = int(row["source_step"])
            activation = int(row["activation_step"])
            activation_steps[source] = activation
            continue
        if "cached_vlm_step" not in row or "control_step" not in row:
            raise ValueError("trace rows require source_step/activation_step or cached_vlm_step/control_step")
        source = row["cached_vlm_step"]
        if source is None:
            continue
        source = int(source)
        if source != previous_source:
            activation_steps.setdefault(source, int(row["control_step"]))
            previous_source = source

    return dict(sorted(_normalize_activation_steps(activation_steps).items()))


def load_trace_max_control_step(path: str | Path) -> int:
    """Return the last control step covered by an evaluator trace."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"alignment trace does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _find_step_trace(payload)
    if rows:
        if any("control_step" in row for row in rows):
            return max(int(row["control_step"]) for row in rows if "control_step" in row)
        if any("activation_step" in row for row in rows):
            return max(
                int(row["activation_step"])
                for row in rows
                if row.get("activation_step") is not None
            )

    def find_max(value: Any) -> int | None:
        if isinstance(value, dict):
            max_steps = value.get("max_control_steps")
            if max_steps is not None:
                return max(0, int(max_steps) - 1)
            for child in value.values():
                found = find_max(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_max(child)
                if found is not None:
                    return found
        return None

    max_control_step = find_max(payload)
    if max_control_step is None:
        raise ValueError(f"no control-step coverage found in alignment trace {path}")
    return max_control_step


def observed_activation_steps(rows: list[dict[str, Any]]) -> dict[int, int]:
    """Extract source activation transitions from a per-control-step trace."""
    activation_steps: dict[int, int] = {}
    previous_source = None
    for row in rows:
        source = row.get("cached_vlm_step")
        if source is None:
            continue
        source = int(source)
        if source != previous_source:
            activation_steps.setdefault(source, int(row["control_step"]))
            previous_source = source
    return activation_steps
