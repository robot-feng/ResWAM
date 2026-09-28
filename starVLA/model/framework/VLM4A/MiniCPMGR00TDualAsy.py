# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""MiniCPM-V + DINO dual-frequency policy.

The VLM consumes a low-rate observation stream while DINO and the action head
consume the current observation on every control step. At inference, VLM refresh
jobs run on a dedicated worker and publish complete hidden-state snapshots;
actions keep using the newest completed snapshot while a refresh is in flight.
Training accepts ``vlm_image`` on each example to align its VLM condition with
the low-rate anchor frame for that control step.
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenDual import Qwen_Dual
from starVLA.model.framework.VLM4A.minicpm_dual_asy_alignment import (
    ALIGNMENT_MODES,
    load_trace_activation_steps,
    source_step_at,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _asy_config(config):
    if config is None:
        raw = {}
    elif isinstance(config, DictConfig):
        raw = OmegaConf.to_container(config, resolve=False)
    elif isinstance(config, dict):
        raw = copy.deepcopy(config)
    else:
        raw = OmegaConf.to_container(OmegaConf.create(config), resolve=False)

    cfg = OmegaConf.create(raw or {})
    cfg.setdefault("framework", {})
    cfg.framework.name = "MiniCPMGR00TDualAsy"
    cfg.framework.setdefault("qwenvl", {})
    local_model = Path("/data/tzq/datasets/starVLA/playground/Pretrained_models/MiniCPM-V-4.6")
    if not cfg.framework.qwenvl.get("base_vlm"):
        cfg.framework.qwenvl.base_vlm = str(local_model) if local_model.is_dir() else "openbmb/MiniCPM-V-4.6"
    if not cfg.framework.qwenvl.get("attn_implementation"):
        cfg.framework.qwenvl.attn_implementation = "sdpa"
    cfg.framework.setdefault("dino", {"dino_backbone": "dinov2_vits14"})
    cfg.framework.setdefault("action_model", {})
    cfg.framework.setdefault("async_alignment", {})
    cfg.framework.async_alignment.setdefault("mode", "wall_clock")
    cfg.framework.async_alignment.setdefault("fixed_latency_steps", None)
    cfg.framework.async_alignment.setdefault("trace_path", None)
    if cfg.framework.get("vlm_refresh_interval") is None:
        cfg.framework.vlm_refresh_interval = 8
    action_defaults = {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "action_horizon": 8,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
    }
    for key, value in action_defaults.items():
        if cfg.framework.action_model.get(key) is None:
            cfg.framework.action_model[key] = value
    if "datasets" not in cfg or cfg.datasets is None:
        cfg.datasets = {}
    if "vla_data" not in cfg.datasets or cfg.datasets.vla_data is None:
        cfg.datasets.vla_data = {"obs_image_size": [224, 224]}
    elif "obs_image_size" not in cfg.datasets.vla_data:
        cfg.datasets.vla_data.obs_image_size = [224, 224]
    return cfg


@FRAMEWORK_REGISTRY.register("MiniCPMGR00TDualAsy")
class MiniCPMGR00TDualAsy(Qwen_Dual):
    """Low-rate MiniCPM-V refresh with per-step DINO and action inference."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        cfg = _asy_config(config)
        super().__init__(config=cfg, **kwargs)
        # Keep VLM refresh cadence separate from action chunk/execution lengths.
        self.vlm_refresh_interval = int(self.config.framework.vlm_refresh_interval)
        self.vlm_update_interval = self.vlm_refresh_interval
        if self.vlm_update_interval < 1:
            raise ValueError("framework.vlm_refresh_interval must be >= 1")

        alignment_cfg = self.config.framework.async_alignment
        self.async_alignment_mode = str(alignment_cfg.mode)
        if self.async_alignment_mode not in ALIGNMENT_MODES:
            raise ValueError(
                f"framework.async_alignment.mode must be one of {ALIGNMENT_MODES}, "
                f"got {self.async_alignment_mode!r}"
            )
        fixed_latency = alignment_cfg.get("fixed_latency_steps")
        self.fixed_latency_steps = None if fixed_latency is None else int(fixed_latency)
        if self.async_alignment_mode == "fixed_step_delay":
            if self.fixed_latency_steps is None or self.fixed_latency_steps < 0:
                raise ValueError(
                    "fixed_step_delay requires nonnegative "
                    "framework.async_alignment.fixed_latency_steps"
                )
        elif self.fixed_latency_steps is not None:
            raise ValueError(
                "framework.async_alignment.fixed_latency_steps is only valid "
                "with fixed_step_delay"
            )
        trace_path = alignment_cfg.get("trace_path")
        self.trace_activation_steps = None
        if self.async_alignment_mode == "trace_replay":
            if not trace_path:
                raise ValueError(
                    "trace_replay requires framework.async_alignment.trace_path"
                )
            self.trace_activation_steps = load_trace_activation_steps(trace_path)
            invalid_sources = [
                source
                for source in self.trace_activation_steps
                if source > 0 and source % self.vlm_refresh_interval != 0
            ]
            if invalid_sources:
                raise ValueError(
                    "trace_replay source steps must match framework.vlm_refresh_interval; "
                    f"invalid sources: {invalid_sources}"
                )

        self._async_generation = 0
        self._control_step = 0
        self._cached_vlm_hidden = None
        self._cached_instruction = None
        self._cached_vlm_step = None
        self._cached_ready_timestamp = None
        self._completed_snapshots = {}
        self._activation_events = []
        self._request_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._worker_stream = None
        self._stats = {
            "vlm_submitted": 0,
            "vlm_completed": 0,
            "vlm_dropped_stale": 0,
            "dino_calls": 0,
            "action_calls": 0,
            "vlm_seconds": [],
        }

    def _encode_vlm(self, vlm_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=vlm_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        layer = self.config.framework.action_model.get("connect_layer_index", -1)
        return outputs.hidden_states[layer].detach()

    def _ensure_worker(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._vlm_worker_loop,
            name="minicpm-vlm-refresh",
            daemon=True,
        )
        self._worker.start()

    def _vlm_worker_loop(self):
        if torch.cuda.is_available():
            try:
                device = next(self.qwen_vl_interface.parameters()).device
                self._worker_stream = torch.cuda.Stream(device=device)
            except StopIteration:
                self._worker_stream = torch.cuda.Stream()

        while True:
            request = self._request_queue.get()
            try:
                if request is None:
                    return
                generation, control_step, images, instructions, request_timestamp = request
                if generation != self._async_generation:
                    self._stats["vlm_dropped_stale"] += 1
                    continue
                started = time.perf_counter()
                with torch.inference_mode():
                    if self._worker_stream is None:
                        hidden = self._encode_vlm(images, instructions)
                    else:
                        with torch.cuda.stream(self._worker_stream):
                            hidden = self._encode_vlm(images, instructions)
                            self._worker_stream.synchronize()
                ready_timestamp = time.perf_counter()
                result = {
                    "generation": generation,
                    "source_step": control_step,
                    "instructions": instructions,
                    "hidden": hidden,
                    "compute_seconds": ready_timestamp - started,
                    "request_timestamp": request_timestamp,
                    "ready_timestamp": ready_timestamp,
                    "error": None,
                }
            except Exception as exc:  # surfaced in the control thread
                ready_timestamp = time.perf_counter()
                result = {
                    "generation": generation,
                    "source_step": control_step,
                    "instructions": instructions,
                    "hidden": None,
                    "compute_seconds": ready_timestamp - started,
                    "request_timestamp": request_timestamp,
                    "ready_timestamp": ready_timestamp,
                    "error": exc,
                }
            finally:
                self._request_queue.task_done()
            self._result_queue.put(result)

    def _record_activation(self, item, activation_timestamp, activation_step=None):
        self._cached_vlm_step = int(item["source_step"])
        self._cached_instruction = item["instructions"]
        self._cached_vlm_hidden = item["hidden"]
        self._cached_ready_timestamp = float(item["ready_timestamp"])
        self._activation_events.append(
            {
                "generation": self._async_generation,
                "source_step": int(item["source_step"]),
                "request_step": int(item["source_step"]),
                "activation_step": (
                    max(0, self._control_step - 1)
                    if activation_step is None
                    else int(activation_step)
                ),
                "request_timestamp": float(item["request_timestamp"]),
                "ready_timestamp": float(item["ready_timestamp"]),
                "activation_timestamp": float(activation_timestamp),
                "compute_seconds": float(item["compute_seconds"]),
                "alignment_mode": self.async_alignment_mode,
            }
        )

    def _poll_vlm_results(self, activation_step=None):
        latest = None
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            source_step = self._accept_vlm_result(item)
            if source_step is not None and (latest is None or source_step >= latest[0]):
                latest = (source_step, item)
        if (
            latest is not None
            and self.async_alignment_mode == "wall_clock"
            and activation_step is not None
        ):
            self._record_activation(
                latest[1], time.perf_counter(), activation_step=activation_step
            )

    def _accept_vlm_result(self, item):
        generation = item["generation"]
        step = int(item["source_step"])
        if generation != self._async_generation:
            self._stats["vlm_dropped_stale"] += 1
            return None
        if item["error"] is not None:
            raise RuntimeError(f"Asynchronous VLM refresh failed at control step {step}") from item["error"]
        self._stats["vlm_completed"] += 1
        self._stats["vlm_seconds"].append(float(item["compute_seconds"]))
        self._completed_snapshots[step] = item
        return step

    def _submit_vlm_refresh(self, images, instructions, control_step):
        # Clone the list containers so callers cannot mutate the queued snapshot.
        self._ensure_worker()
        request_timestamp = time.perf_counter()
        self._request_queue.put(
            (
                self._async_generation,
                int(control_step),
                copy.deepcopy(images),
                tuple(instructions),
                request_timestamp,
            )
        )
        self._stats["vlm_submitted"] += 1

    def _run_synchronous_refresh(self, images, instructions, control_step):
        request_timestamp = time.perf_counter()
        self._stats["vlm_submitted"] += 1
        with torch.inference_mode():
            hidden = self._encode_vlm(images, instructions)
        ready_timestamp = time.perf_counter()
        self._stats["vlm_completed"] += 1
        elapsed = ready_timestamp - request_timestamp
        self._stats["vlm_seconds"].append(elapsed)
        self._record_activation(
            {
                "source_step": int(control_step),
                "instructions": tuple(instructions),
                "hidden": hidden,
                "request_timestamp": request_timestamp,
                "ready_timestamp": ready_timestamp,
                "compute_seconds": elapsed,
            },
            ready_timestamp,
            activation_step=control_step,
        )

    def _scheduled_source_step(self, control_step):
        if self.async_alignment_mode == "fixed_step_delay":
            return source_step_at(
                control_step,
                mode=self.async_alignment_mode,
                refresh_interval=self.vlm_refresh_interval,
                fixed_latency_steps=self.fixed_latency_steps,
            )
        if self.async_alignment_mode == "trace_replay":
            return source_step_at(
                control_step,
                mode=self.async_alignment_mode,
                refresh_interval=self.vlm_refresh_interval,
                trace_activation_steps=self.trace_activation_steps,
            )
        raise RuntimeError("scheduled source is only defined for controlled alignment modes")

    def _wait_for_scheduled_source(self, source_step, activation_step, timeout=120.0):
        deadline = time.monotonic() + timeout
        while source_step not in self._completed_snapshots:
            self._poll_vlm_results()
            if source_step in self._completed_snapshots:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for scheduled VLM source {source_step}"
                )
            try:
                item = self._result_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            self._accept_vlm_result(item)
        if self._cached_vlm_step != source_step:
            self._record_activation(
                self._completed_snapshots[source_step],
                time.perf_counter(),
                activation_step=activation_step,
            )

    def reset_async_cache(self):
        """Start a new episode and invalidate pending outputs from the old one."""
        self._async_generation += 1
        self._control_step = 0
        self._cached_vlm_hidden = None
        self._cached_instruction = None
        self._cached_vlm_step = None
        self._cached_ready_timestamp = None
        self._completed_snapshots.clear()
        self._activation_events.clear()
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def close_async_worker(self, timeout: float = 30.0):
        """Stop the background worker cleanly after evaluation."""
        if self._worker is None:
            return
        self._request_queue.put(None)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError("MiniCPM VLM refresh worker did not stop in time")
        self._poll_vlm_results()
        self._worker = None

    def wait_for_async_refreshes(self, timeout: float = 30.0):
        """Wait for submitted refreshes and surface completed worker errors."""
        deadline = time.monotonic() + timeout
        while True:
            self._poll_vlm_results()
            with self._request_queue.all_tasks_done:
                pending = self._request_queue.unfinished_tasks
            if pending == 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out with {pending} asynchronous VLM refreshes pending"
                )
            time.sleep(0.01)
        self._poll_vlm_results()

    def async_stats(self):
        result = {k: v for k, v in self._stats.items() if k != "vlm_seconds"}
        durations = self._stats["vlm_seconds"]
        result["vlm_mean_seconds"] = float(np.mean(durations)) if durations else 0.0
        result["latest_vlm_step"] = self._cached_vlm_step
        result["alignment_mode"] = self.async_alignment_mode
        result["vlm_activation_events"] = list(self._activation_events)
        result["latest_vlm_activation_event"] = (
            dict(self._activation_events[-1]) if self._activation_events else None
        )
        result["vlm_age_steps"] = (
            max(0, self._control_step - 1 - self._cached_vlm_step)
            if self._cached_vlm_step is not None
            else None
        )
        result["queued_refreshes"] = self._request_queue.qsize()
        return result

    def _condition_from_vlm_hidden(self, batch_images, wrist_views, state, vlm_hidden):
        if wrist_views is None:
            wrist_views = batch_images
        image_tensors = self.dino_encoder.prepare_dino_input(wrist_views)
        batch_size = len(batch_images)
        dino_features = self.dino_encoder(image_tensors)
        dino_features = dino_features.reshape(batch_size, -1, dino_features.shape[-1])
        dino_features = self.dino_pro(dino_features)
        last_hidden = torch.cat([vlm_hidden, dino_features], dim=1)
        self._stats["dino_calls"] += 1
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        return last_hidden, state

    def forward(self, examples=None, **kwargs):
        """Train from the current DINO image and an explicitly aligned VLM frame.

        Each example can include ``vlm_image`` (a list of camera views sampled
        at the most recent VLM refresh). If omitted, current ``image`` is used.
        """
        if not examples:
            raise ValueError("MiniCPMGR00TDualAsy.forward requires at least one example")
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
        vlm_images = [
            to_pil_preserve(example.get("vlm_image", example["image"]))
            for example in examples
        ]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", [224, 224])
        if train_obs_image_size:
            vlm_images = resize_images(vlm_images, target_size=train_obs_image_size)

        vlm_hidden = self._encode_vlm(vlm_images, instructions)
        last_hidden, state = self._condition_from_vlm_hidden(batch_images, wrist_views, state, vlm_hidden)
        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )[:, -self.action_horizon :, :]
        repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        with torch.autocast("cuda", dtype=torch.float32, enabled=torch.cuda.is_available()):
            loss = self.action_model(
                last_hidden.repeat(repeats, 1, 1),
                actions.repeat(repeats, 1, 1),
                state.repeat(repeats, 1, 1) if state is not None else None,
            )
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples=None, control_step: Optional[int] = None, **kwargs):
        if examples is None:
            raise ValueError("MiniCPMGR00TDualAsy.predict_action requires examples")
        if not isinstance(examples, list):
            examples = [examples]
        if len(examples) != 1:
            raise ValueError("Stateful DualAsy inference currently requires batch size 1")

        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
        instruction_key = tuple(instructions)
        if self._cached_instruction is not None and self._cached_instruction != instruction_key:
            self.reset_async_cache()

        step = self._control_step if control_step is None else int(control_step)
        self._control_step = max(self._control_step, step + 1)
        refresh_due = self._cached_vlm_hidden is None or step % self.vlm_update_interval == 0
        if refresh_due:
            if self.async_alignment_mode == "synchronous":
                self._run_synchronous_refresh(batch_images, instructions, step)
            else:
                if self.async_alignment_mode == "trace_replay" and step not in self.trace_activation_steps:
                    raise ValueError(
                        f"trace_replay has no activation event for requested source step {step}"
                    )
                self._submit_vlm_refresh(batch_images, instructions, step)

        self._poll_vlm_results(activation_step=step)
        if self.async_alignment_mode in ("fixed_step_delay", "trace_replay"):
            scheduled_source = self._scheduled_source_step(step)
            self._wait_for_scheduled_source(scheduled_source, activation_step=step)
        elif self._cached_vlm_hidden is None:
            # The bootstrap refresh is awaited in every mode; after bootstrap,
            # only the wall-clock mode keeps using the latest completed cache.
            deadline = time.monotonic() + 120.0
            while self._cached_vlm_hidden is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for the first MiniCPM VLM snapshot")
                try:
                    item = self._result_queue.get(timeout=min(remaining, 0.1))
                except queue.Empty:
                    continue
                else:
                    self._accept_vlm_result(item)
                    if self.async_alignment_mode == "wall_clock":
                        self._record_activation(
                            item, time.perf_counter(), activation_step=step
                        )

        condition, state = self._condition_from_vlm_hidden(batch_images, wrist_views, state, self._cached_vlm_hidden)
        with torch.autocast("cuda", dtype=torch.float32, enabled=torch.cuda.is_available()):
            actions = self.action_model.predict_action(condition, state)
        self._stats["action_calls"] += 1
        return {
            "normalized_actions": actions.detach().cpu().numpy(),
            "async_stats": self.async_stats(),
        }
