#!/usr/bin/env python3
"""Dedicated single-process trainer for MiniCPMGR00TRes.

It deliberately bypasses starVLA's action-only trainer so residual and optional
assistant-text losses are logged and backpropagated explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.minicpm_res_lerobot import MiniCPMResidualLeRobotDataset
from starVLA.model.framework.VLM4A.MiniCPMGR00TRes import MiniCPMGR00TRes


def _collate(batch):
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("minicpm_gr00t_res_libero.yaml"),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None, help="checkpoint cadence override")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="initialize trainable weights from a strict ResWAM checkpoint; optimizer state starts fresh",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-ids", default=None, help="optional comma-separated episode IDs for a controlled subset")
    parser.add_argument("--control-steps", default=None, help="optional comma-separated control-step indices")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and run one loss/gradient step")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    inherited = cfg.get("defaults")
    if inherited:
        if not OmegaConf.is_list(inherited) or len(inherited) != 1 or not isinstance(inherited[0], str):
            raise ValueError("only one string config parent is supported in defaults")
        parent_path = (args.config.parent / inherited[0]).with_suffix(".yaml")
        if not parent_path.is_file():
            raise FileNotFoundError(f"config parent not found: {parent_path}")
        del cfg["defaults"]
        cfg = OmegaConf.merge(OmegaConf.load(parent_path), cfg)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    dc = cfg.datasets.residual_data
    episode_ids = (
        None if args.episode_ids is None else [item.strip() for item in args.episode_ids.split(",") if item.strip()]
    )
    control_steps = (
        None
        if args.control_steps is None
        else [int(item.strip()) for item in args.control_steps.split(",") if item.strip()]
    )
    dataset = MiniCPMResidualLeRobotDataset(
        data_root_dir=dc.data_root_dir,
        dataset_name=dc.dataset_name,
        robot_type=dc.robot_type,
        success_terminal_manifest=dc.success_terminal_manifest,
        assistant_label_manifest=dc.get("assistant_label_manifest"),
        camera_key=dc.camera_key,
        video_backend=dc.video_backend,
        data_cfg={"video_backend": dc.video_backend, "include_state": False},
        max_history_frames=dc.get("max_history_frames"),
        sample_stride=int(cfg.framework.runtime.vlm_refresh_interval),
        episode_ids=episode_ids,
        control_steps=control_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.trainer.get("per_device_batch_size", 1)),
        # Samples are ordered by episode so the adapter decodes each episode's
        # frames once instead of repeatedly reloading them under random shuffles.
        shuffle=False,
        num_workers=int(cfg.trainer.get("num_workers", 0)),
        collate_fn=_collate,
        drop_last=False,
    )
    model = MiniCPMGR00TRes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.init_checkpoint is not None:
        model.load_reswam_checkpoint(args.init_checkpoint)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("MiniCPMGR00TRes has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    output_dir = args.output_dir or Path(cfg.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_steps or cfg.trainer.max_train_steps)
    grad_accum = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    log_every = int(cfg.trainer.get("log_every", 10))
    save_every = int(args.save_every if args.save_every is not None else cfg.trainer.get("save_every", 100))
    max_grad_norm = float(cfg.trainer.get("max_grad_norm", 1.0))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    records_path = output_dir / "train_metrics.jsonl"
    started = time.perf_counter()
    update = 0
    micro_step = 0
    while update < max_steps:
        for batch in loader:
            micro_step += 1
            outputs = model(batch)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite total loss at optimizer step {update + 1}")
            (loss / grad_accum).backward()
            if micro_step % grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at optimizer step {update + 1}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            row = {
                "optimizer_step": update,
                "residual_loss": float(outputs["residual_loss"].detach()),
                "text_loss": float(outputs["text_loss"].detach()),
                "text_supervision_enabled": bool(outputs["text_supervision_enabled"].item()),
                "total_loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if update % log_every == 0 or update == 1:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.dry_run or (save_every > 0 and update % save_every == 0):
                model.save_reswam_checkpoint(output_dir / f"step_{update:08d}")
            if args.dry_run or update >= max_steps:
                break
        else:
            continue
        if args.dry_run or update >= max_steps:
            break
    if update and (save_every <= 0 or update % save_every):
        model.save_reswam_checkpoint(output_dir / "final")
    print(f"Finished {update} optimizer steps; output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
