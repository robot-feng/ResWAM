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
    action_defaults = {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "action_horizon": 8,
        # Number of low-level control steps between upper VLM refreshes.
        # It is independent from action_horizon (the predicted action chunk).
        "execution_horizon": None,
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
        # action_horizon is the predicted action chunk length. execution_horizon
        # is the number of low-level control steps per upper-system refresh.
        # Older configs without execution_horizon retain their former cadence.
        configured_execution_horizon = self.config.framework.action_model.get(
            "execution_horizon"
        )
        self.execution_horizon = int(
            configured_execution_horizon
            if configured_execution_horizon is not None
            else self.action_horizon
        )
        self.vlm_update_interval = self.execution_horizon
        if self.vlm_update_interval < 1:
            raise ValueError("framework.action_model.execution_horizon must be >= 1")

        self._async_generation = 0
        self._control_step = 0
        self._cached_vlm_hidden = None
        self._cached_instruction = None
        self._cached_vlm_step = None
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
                generation, control_step, images, instructions = request
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
                result = (generation, control_step, instructions, hidden, time.perf_counter() - started, None)
            except Exception as exc:  # surfaced in the control thread
                result = (generation, control_step, instructions, None, time.perf_counter() - started, exc)
            finally:
                self._request_queue.task_done()
            self._result_queue.put(result)

    def _poll_vlm_results(self):
        latest = None
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            generation, step, instructions, hidden, elapsed, error = item
            if generation != self._async_generation:
                self._stats["vlm_dropped_stale"] += 1
                continue
            if error is not None:
                raise RuntimeError(f"Asynchronous VLM refresh failed at control step {step}") from error
            self._stats["vlm_completed"] += 1
            self._stats["vlm_seconds"].append(elapsed)
            if latest is None or step >= latest[0]:
                latest = (step, instructions, hidden)
        if latest is not None:
            self._cached_vlm_step, self._cached_instruction, self._cached_vlm_hidden = latest

    def _submit_vlm_refresh(self, images, instructions, control_step):
        # Clone the list containers so callers cannot mutate the queued snapshot.
        self._ensure_worker()
        self._request_queue.put(
            (self._async_generation, int(control_step), copy.deepcopy(images), tuple(instructions))
        )
        self._stats["vlm_submitted"] += 1

    def reset_async_cache(self):
        """Start a new episode and invalidate pending outputs from the old one."""
        self._async_generation += 1
        self._control_step = 0
        self._cached_vlm_hidden = None
        self._cached_instruction = None
        self._cached_vlm_step = None
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
        if self._cached_vlm_hidden is None or step % self.vlm_update_interval == 0:
            self._submit_vlm_refresh(batch_images, instructions, step)

        self._poll_vlm_results()
        if self._cached_vlm_hidden is None:
            # The first action in an episode must have a valid semantic context.
            deadline = time.monotonic() + 120.0
            while self._cached_vlm_hidden is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for the first MiniCPM VLM snapshot")
                try:
                    item = self._result_queue.get(timeout=min(remaining, 0.1))
                except queue.Empty:
                    continue
                generation, task_step, task_instructions, hidden, elapsed, error = item
                if generation != self._async_generation:
                    self._stats["vlm_dropped_stale"] += 1
                    continue
                if error is not None:
                    raise RuntimeError(f"Asynchronous VLM refresh failed at control step {task_step}") from error
                self._stats["vlm_completed"] += 1
                self._stats["vlm_seconds"].append(elapsed)
                self._cached_vlm_step = task_step
                self._cached_instruction = task_instructions
                self._cached_vlm_hidden = hidden

        condition, state = self._condition_from_vlm_hidden(batch_images, wrist_views, state, self._cached_vlm_hidden)
        with torch.autocast("cuda", dtype=torch.float32, enabled=torch.cuda.is_available()):
            actions = self.action_model.predict_action(condition, state)
        self._stats["action_calls"] += 1
        return {
            "normalized_actions": actions.detach().cpu().numpy(),
            "async_stats": self.async_stats(),
        }
