#!/usr/bin/env python3
"""Exercise prefix-cache validation on a synthetic MiniCPM video history."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.MiniCPMGR00TRes import MiniCPMGR00TRes


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_res_libero.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("reswam_prefix_cache_smoke.json"))
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")

    cfg = _load_cfg(args.config)
    cfg.framework.runtime.cache_mode = "prefix"
    # Compare both the initial prefill and each subsequent append in this smoke.
    cfg.framework.runtime.prefix_cache_validate_every = 1
    model = MiniCPMGR00TRes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if args.checkpoint is not None:
        model.load_reswam_checkpoint(args.checkpoint)

    rows = []
    t0 = time.perf_counter()
    horizon = model.vlm_refresh_interval
    for frame_idx in range(args.frames):
        image = Image.new(
            "RGB",
            (224, 224),
            (31 + (frame_idx * 17) % 190, 47 + (frame_idx * 29) % 180, 83 + (frame_idx * 11) % 160),
        )
        result = model.observe(
            image=image,
            control_step=frame_idx,
            timestamp_seconds=frame_idx / model.history_adapter.fps,
            instruction="move the mug to the plate",
            episode_id="synthetic-cache-smoke",
        )
        _sync(device)
        rows.append(
            {
                "control_step": frame_idx,
                "refresh_step": result.get("control_step"),
                "refreshed": result.get("refreshed"),
                "plan_age": result.get("plan_age"),
                "history_frame_count": result.get("history_frame_count"),
                "plan_history_frame_count": result.get("plan_history_frame_count"),
                "plan_control_step": result.get("plan_control_step"),
                "cache_mode": result.get("cache_mode"),
                "cache_validated": result.get("cache_validated"),
                "cache_context_validated": result.get("cache_context_validated"),
                "cache_relative_rms_error": result.get("cache_relative_rms_error"),
                "cache_mean_absolute_error": result.get("cache_mean_absolute_error"),
                "cache_max_absolute_error": result.get("cache_max_absolute_error"),
                "cache_fallback_reason": result.get("cache_fallback_reason"),
            }
        )

    report = {
        "framework": "MiniCPMGR00TRes",
        "device": str(device),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "synthetic_frames": args.frames,
        "vlm_refresh_interval": horizon,
        "cache_validate_every": model.prefix_cache_validate_every,
        "cache_max_relative_rms": model.prefix_cache_max_relative_rms,
        "elapsed_seconds": time.perf_counter() - t0,
        "cache_validated": model._prefix_cache_validated,
        "cache_disabled_reason": model._prefix_cache_disabled_reason,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
