#!/usr/bin/env python3
"""Profile full-history inference on deterministic synthetic video histories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.MiniCPMGR00TRes import MiniCPMGR00TRes
from starVLA.model.framework.VLM4A.minicpm_video_history import HistoryFrame, gather_token_hidden_states


def _load_cfg(path: Path):
    cfg = OmegaConf.load(path)
    inherited = cfg.get("defaults")
    if inherited:
        if len(inherited) != 1 or not isinstance(inherited[0], str):
            raise ValueError("only one string config parent is supported in defaults")
        parent_path = (path.parent / inherited[0]).with_suffix(".yaml")
        del cfg["defaults"]
        cfg = OmegaConf.merge(OmegaConf.load(parent_path), cfg)
    return cfg


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _training_smoke(model: MiniCPMGR00TRes, device: torch.device) -> dict:
    current = Image.new("RGB", (224, 224), (40, 90, 150))
    terminal = Image.new("RGB", (224, 224), (160, 60, 25))
    frame = HistoryFrame(current, "synthetic-train", 0, 0.0, "primary", 0)
    example = {
        "lang": "move the mug to the plate",
        "episode_id": "synthetic-train",
        "control_step": 0,
        "timestamp_seconds": 0.0,
        "history": [frame],
        "current_image": current,
        "terminal_image": terminal,
        "assistant_text": "The mug should be moved to the plate.",
    }
    model.train()
    model.zero_grad(set_to_none=True)
    baseline_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = model([example])
    _sync(device)
    forward_seconds = time.perf_counter() - start
    result["loss"].backward()
    _sync(device)
    total_seconds = time.perf_counter() - start
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    nonzero_grad_tensors = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        and parameter.grad.abs().sum().item() > 0
        for parameter in trainable
    )
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    profile = {
        "status": "ok",
        "data": "one synthetic image and one synthetic assistant label; no optimizer step",
        "residual_loss": float(result["residual_loss"].detach()),
        "text_loss": float(result["text_loss"].detach()),
        "total_loss": float(result["loss"].detach()),
        "forward_seconds": forward_seconds,
        "forward_and_backward_seconds": total_seconds,
        "trainable_tensor_count": len(trainable),
        "nonzero_finite_grad_tensor_count": int(nonzero_grad_tensors),
        "peak_cuda_memory_bytes": peak_memory,
        "model_baseline_cuda_memory_bytes": baseline_memory if device.type == "cuda" else None,
        "transient_peak_cuda_memory_bytes": (
            max(0, peak_memory - baseline_memory) if peak_memory is not None else None
        ),
        "optimizer_step_performed": False,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_res_libero.yaml"),
    )
    parser.add_argument("--history-lengths", default="1,8,32")
    parser.add_argument("--training-smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reswam_synthetic_profile.json"))
    args = parser.parse_args()
    lengths = [int(item.strip()) for item in args.history_lengths.split(",") if item.strip()]
    if not lengths or min(lengths) < 1:
        raise ValueError("--history-lengths must be a comma-separated list of positive frame counts")

    cfg = _load_cfg(args.config)
    cfg.framework.runtime.cache_mode = "recompute"
    cfg.framework.residual_model.history_mode = "full"
    model = MiniCPMGR00TRes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = []
    instruction = "move the mug to the plate"
    for count in lengths:
        frames = [
            HistoryFrame(
                image=Image.new(
                    "RGB",
                    (224, 224),
                    (31 + (index * 17) % 190, 47 + (index * 29) % 180, 83 + (index * 11) % 160),
                ),
                episode_id="synthetic-profile",
                control_step=index * model.execution_horizon,
                timestamp_seconds=index / model.history_adapter.fps,
                view_id="primary",
                frame_order=index,
            )
            for index in range(count)
        ]
        model.reset_episode("synthetic-profile", instruction)
        model._history = frames
        model._control_step = frames[-1].control_step
        if device.type == "cuda":
            torch.cuda.empty_cache()
            baseline_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            free_memory_before, _ = torch.cuda.mem_get_info(device)
        else:
            baseline_memory = 0
            free_memory_before = None

        inputs = outputs = query = prediction = reference = None
        try:
            t0 = time.perf_counter()
            inputs = model.history_adapter.tokenize_history_query(
                instruction,
                "synthetic-profile",
                frames,
                history_mode="full",
                device=device,
            )
            _sync(device)
            preprocess_seconds = time.perf_counter() - t0
            context_tokens = model._check_context_budget(inputs)

            t0 = time.perf_counter()
            outputs = model._encode_vlm(inputs, use_cache=False)
            _sync(device)
            vlm_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            query = gather_token_hidden_states(
                outputs.last_hidden_state,
                inputs["input_ids"],
                model.residual_token_id,
                expected_count=model.num_residual_tokens,
                attention_mask=inputs.get("attention_mask"),
            )
            prediction = model.residual_head(query.float())
            _sync(device)
            query_head_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            reference = model.encode_dino(frames[-1].image)
            _sync(device)
            dino_seconds = time.perf_counter() - t0
            peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            rows.append(
                {
                    "history_frames": count,
                    "status": "ok",
                    "context_tokens": context_tokens,
                    "residual_shape": list(prediction.shape),
                    "preprocess_seconds": preprocess_seconds,
                    "vlm_seconds": vlm_seconds,
                    "query_and_head_seconds": query_head_seconds,
                    "current_dino_seconds": dino_seconds,
                    "inference_seconds": preprocess_seconds + vlm_seconds + query_head_seconds + dino_seconds,
                    "peak_cuda_memory_bytes": peak_memory,
                    "model_baseline_cuda_memory_bytes": baseline_memory if device.type == "cuda" else None,
                    "transient_peak_cuda_memory_bytes": (
                        max(0, peak_memory - baseline_memory) if peak_memory is not None else None
                    ),
                    "free_gpu_bytes_before": free_memory_before,
                    "reference_feature_shape": list(reference.shape),
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            rows.append(
                {
                    "history_frames": count,
                    "status": "oom",
                    "context_tokens": (
                        model._token_count(inputs) if inputs is not None else None
                    ),
                    "free_gpu_bytes_before": free_memory_before,
                    "peak_cuda_memory_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    ),
                    "error": str(exc).splitlines()[0],
                }
            )
        finally:
            del inputs, outputs, query, prediction, reference
            if device.type == "cuda":
                torch.cuda.empty_cache()

    training_smoke = None
    if args.training_smoke:
        try:
            training_smoke = _training_smoke(model, device)
        except torch.cuda.OutOfMemoryError as exc:
            training_smoke = {
                "status": "oom",
                "error": str(exc).splitlines()[0],
                "optimizer_step_performed": False,
            }
    report = {
        "framework": "MiniCPMGR00TRes",
        "data": "deterministic synthetic images; not a task-performance evaluation",
        "device": str(device),
        "cache_mode": "recompute",
        "batch_size": 1,
        "context_token_budget": model.max_context_tokens,
        "history_profiles": rows,
        "training_smoke": training_smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
