# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LeRobot sample alignment for MiniCPMGR00TDualAsy training."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from starVLA.model.framework.VLM4A.minicpm_dual_asy_alignment import (
    ALIGNMENT_MODES,
    fixed_activation_steps,
    load_trace_activation_steps,
    load_trace_max_control_step,
    source_step_at,
)


def _to_plain_dict(value: Any) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=False)
    elif not isinstance(value, dict):
        value = OmegaConf.to_container(OmegaConf.create(value or {}), resolve=False)
    return dict(value or {})


def resolve_async_training_alignment(framework_cfg: Any) -> dict:
    """Resolve offline anchor construction from a DualAsy framework config.

    Wall-clock readiness cannot be reconstructed from the current minibatch.
    It therefore requires a previous trace and trains with that trace replayed.
    """
    framework = _to_plain_dict(framework_cfg)
    alignment = _to_plain_dict(framework.get("async_alignment", {}))
    runtime_mode = str(alignment.get("mode", "wall_clock"))
    if runtime_mode not in ALIGNMENT_MODES:
        raise ValueError(
            f"framework.async_alignment.mode must be one of {ALIGNMENT_MODES}, "
            f"got {runtime_mode!r}"
        )

    refresh_interval = int(framework.get("vlm_refresh_interval", 8))
    if refresh_interval < 1:
        raise ValueError("framework.vlm_refresh_interval must be >= 1")
    execution_horizon = int(framework.get("execution_horizon", 1))
    if execution_horizon < 1:
        raise ValueError("framework.execution_horizon must be >= 1")

    fixed_latency_steps = alignment.get("fixed_latency_steps")
    if runtime_mode == "fixed_step_delay":
        if fixed_latency_steps is None or int(fixed_latency_steps) < 0:
            raise ValueError(
                "fixed_step_delay training requires nonnegative "
                "framework.async_alignment.fixed_latency_steps"
            )
        fixed_latency_steps = int(fixed_latency_steps)
        training_mode = runtime_mode
        trace_path = None
    elif runtime_mode in ("trace_replay", "wall_clock"):
        runtime_trace_path = alignment.get("trace_path")
        if runtime_mode == "wall_clock":
            trace_path = alignment.get("training_trace_path")
        else:
            trace_path = alignment.get("training_trace_path") or runtime_trace_path
        if not trace_path:
            flag = "training_trace_path" if runtime_mode == "wall_clock" else "trace_path"
            raise ValueError(
                f"{runtime_mode} LeRobot training requires "
                f"framework.async_alignment.{flag}"
            )
        trace_path = str(Path(trace_path).expanduser().resolve())
        if runtime_mode == "trace_replay":
            if not runtime_trace_path:
                raise ValueError("trace_replay requires framework.async_alignment.trace_path")
            runtime_trace_path = str(Path(runtime_trace_path).expanduser().resolve())
            if trace_path != runtime_trace_path:
                raise ValueError(
                    "trace_replay training and runtime must use the same activation trace"
                )
        training_mode = "trace_replay"
        fixed_latency_steps = None
    else:
        training_mode = "synchronous"
        trace_path = None
        fixed_latency_steps = None

    trace_activation_steps = (
        load_trace_activation_steps(trace_path) if trace_path else None
    )
    trace_max_control_step = (
        load_trace_max_control_step(trace_path) if trace_path else None
    )
    return {
        "mode": training_mode,
        "runtime_mode": runtime_mode,
        "refresh_interval": refresh_interval,
        "execution_horizon": execution_horizon,
        "fixed_latency_steps": fixed_latency_steps,
        "trace_activation_steps": trace_activation_steps,
        "trace_max_control_step": trace_max_control_step,
        "trace_path": trace_path,
    }


class MiniCPMAsyncTemporalSampler:
    """Attach the latest available same-episode VLM image to a LeRobot sample."""

    def __init__(self, alignment: dict, *, cache_size: int = 0):
        self.mode = str(alignment["mode"])
        self.refresh_interval = int(alignment["refresh_interval"])
        self.execution_horizon = int(alignment.get("execution_horizon", 1))
        self.fixed_latency_steps = alignment.get("fixed_latency_steps")
        if self.fixed_latency_steps is not None:
            self.fixed_latency_steps = int(self.fixed_latency_steps)
        trace_steps = alignment.get("trace_activation_steps")
        self.trace_activation_steps = (
            {int(source): int(activation) for source, activation in trace_steps.items()}
            if trace_steps
            else None
        )
        self.trace_max_control_step = alignment.get("trace_max_control_step")
        if self.trace_max_control_step is not None:
            self.trace_max_control_step = int(self.trace_max_control_step)
        self.cache_size = max(0, int(cache_size))
        self._anchor_image_cache: OrderedDict[tuple[Any, int], Any] = OrderedDict()

        if self.mode not in ("synchronous", "fixed_step_delay", "trace_replay"):
            raise ValueError(
                "offline temporal sampler mode must resolve to synchronous, "
                f"fixed_step_delay, or trace_replay; got {self.mode!r}"
            )
        if self.refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be >= 1")
        if self.mode == "fixed_step_delay" and self.fixed_latency_steps is None:
            raise ValueError("fixed_step_delay requires fixed_latency_steps")
        if self.mode == "trace_replay" and not self.trace_activation_steps:
            raise ValueError("trace_replay requires source_step -> activation_step events")
        if self.mode == "trace_replay":
            invalid_sources = [
                source
                for source in self.trace_activation_steps
                if source > 0 and source % self.refresh_interval != 0
            ]
            if invalid_sources:
                raise ValueError(
                    "trace source steps do not match refresh_interval="
                    f"{self.refresh_interval}: {invalid_sources}"
                )

    def _source_and_activation(self, control_step: int) -> tuple[int, int]:
        control_step = int(control_step)
        self.validate_trace_coverage(control_step)

        source_step = source_step_at(
            control_step,
            mode=self.mode,
            refresh_interval=self.refresh_interval,
            fixed_latency_steps=self.fixed_latency_steps,
            trace_activation_steps=self.trace_activation_steps,
        )
        if self.mode == "trace_replay":
            ready_boundary = self.trace_activation_steps[source_step]
            activation_step = (
                (ready_boundary + self.execution_horizon - 1)
                // self.execution_horizon
            ) * self.execution_horizon
        else:
            latency = 0 if self.mode == "synchronous" else self.fixed_latency_steps
            events = fixed_activation_steps(
                self.refresh_interval,
                int(latency),
                control_step,
                self.execution_horizon,
            )
            activation_step = events[source_step]
        return source_step, activation_step

    def validate_trace_coverage(self, control_step: int, *, context="sample") -> None:
        control_step = int(control_step)
        if (
            self.mode == "trace_replay"
            and self.trace_max_control_step is not None
            and control_step > self.trace_max_control_step
        ):
            raise ValueError(
                f"{context} control_step exceeds alignment trace coverage: "
                f"{control_step} > {self.trace_max_control_step}"
            )

    def align_sample(
        self,
        sample: dict,
        *,
        control_step: int,
        load_anchor_images: Callable[[int], Any],
        cache_key: Any,
    ) -> dict:
        """Preserve current observation/action; add the correct past VLM image."""
        current_step = int(control_step)
        if current_step % self.execution_horizon != 0:
            raise ValueError(
                "DualAsy training samples must be policy-call boundaries: "
                f"control_step={current_step}, K={self.execution_horizon}"
            )
        source_step, activation_step = self._source_and_activation(current_step)
        key = (cache_key, source_step)

        if source_step == current_step:
            anchor_images = sample["image"]
            self._cache_images(key, anchor_images)
        elif key in self._anchor_image_cache:
            anchor_images = self._anchor_image_cache[key]
            self._anchor_image_cache.move_to_end(key)
        else:
            anchor_images = load_anchor_images(source_step)
            self._cache_images(key, anchor_images)

        aligned = dict(sample)
        aligned["vlm_image"] = copy.deepcopy(anchor_images)
        aligned["vlm_source_step"] = source_step
        aligned["vlm_anchor_frame"] = source_step  # pilot compatibility
        aligned["vlm_request_step"] = source_step
        aligned["vlm_activation_step"] = activation_step
        aligned["vlm_current_step"] = current_step
        aligned["vlm_age_steps"] = current_step - source_step
        aligned["vlm_delivery_latency_steps"] = activation_step - source_step
        aligned["vlm_alignment_mode"] = self.mode
        return aligned

    def _cache_images(self, key: tuple[Any, int], images: Any) -> None:
        if self.cache_size <= 0:
            return
        self._anchor_image_cache[key] = copy.deepcopy(images)
        self._anchor_image_cache.move_to_end(key)
        while len(self._anchor_image_cache) > self.cache_size:
            self._anchor_image_cache.popitem(last=False)
