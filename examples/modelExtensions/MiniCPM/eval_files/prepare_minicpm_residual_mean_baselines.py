#!/usr/bin/env python3
"""Build train-only global and task-conditioned mean terminal residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.minicpm_res_lerobot import MiniCPMResidualLeRobotDataset
from starVLA.dataloader.minicpm_res_splits import (
    load_residual_episode_split,
    sha256_file,
)
from starVLA.model.modules.dino_model.dino import get_dino_model


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_res_libero.yaml"
DEFAULT_SPLIT = ROOT / "examples/modelExtensions/MiniCPM/annotations/libero_goal_residual_split_v1.json"
DEFAULT_OUTPUT = ROOT / "playground/Checkpoints/minicpm_res_stage3_target_ablation/train_mean_residuals.npz"


def _equal_weight_step_mean(
    sums: Mapping[int, np.ndarray], counts: Mapping[int, int], steps: list[int]
) -> np.ndarray:
    """Average per-step means so long trajectories do not dominate the baseline."""
    if not steps:
        raise ValueError("at least one control step is required")
    means = []
    for step in steps:
        count = int(counts.get(step, 0))
        if count <= 0 or step not in sums:
            raise ValueError(f"control step {step} has no residual samples")
        means.append(sums[step] / count)
    return np.stack(means, axis=0).mean(axis=0)


def _load_cfg(path: Path):
    cfg = OmegaConf.load(path)
    inherited = cfg.get("defaults")
    if inherited:
        if not OmegaConf.is_list(inherited) or len(inherited) != 1 or not isinstance(inherited[0], str):
            raise ValueError("only one string config parent is supported in defaults")
        parent_path = (path.parent / inherited[0]).with_suffix(".yaml")
        if not parent_path.is_file():
            raise FileNotFoundError(f"config parent not found: {parent_path}")
        del cfg["defaults"]
        cfg = OmegaConf.merge(OmegaConf.load(parent_path), cfg)
    return cfg


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _git_info() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split", choices=("train",), default="train")
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument("--control-step", type=int, help="one control position (legacy form)")
    step_group.add_argument(
        "--control-steps",
        help="comma-separated control positions; global/task means weight each position equally",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.control_steps is not None:
        control_steps = [int(item.strip()) for item in args.control_steps.split(",") if item.strip()]
    else:
        control_steps = [8 if args.control_step is None else int(args.control_step)]
    if not control_steps or min(control_steps) < 0 or len(control_steps) != len(set(control_steps)):
        raise ValueError("control steps must be unique nonnegative indices")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen baseline artifact: {args.output}")
    cfg = _load_cfg(args.config)
    data_cfg = cfg.datasets.residual_data
    split_info = load_residual_episode_split(
        args.split_manifest,
        split=args.split,
        success_terminal_manifest=data_cfg.success_terminal_manifest,
        dataset_path=Path(data_cfg.data_root_dir) / data_cfg.dataset_name,
    )
    dataset = MiniCPMResidualLeRobotDataset(
        data_root_dir=data_cfg.data_root_dir,
        dataset_name=data_cfg.dataset_name,
        robot_type=data_cfg.robot_type,
        success_terminal_manifest=data_cfg.success_terminal_manifest,
        assistant_label_manifest=data_cfg.get("assistant_label_manifest"),
        camera_key=data_cfg.camera_key,
        video_backend=data_cfg.video_backend,
        data_cfg={"video_backend": data_cfg.video_backend, "include_state": False},
        history_mode="current_only",
        sample_stride=int(cfg.framework.runtime.vlm_refresh_interval),
        episode_ids=split_info["episode_ids"],
        control_steps=control_steps,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = get_dino_model(str(cfg.framework.dino.dino_backbone)).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    teacher_sha256 = _module_sha256(dino)

    global_sums: dict[int, np.ndarray] = {}
    global_counts: dict[int, int] = defaultdict(int)
    task_sums: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    task_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    episode_ids: set[str] = set()
    entry_by_id = {str(row["episode_id"]): row for row in split_info["entries"]}
    for index in range(len(dataset)):
        sample = dataset[index]
        episode_id = str(sample["episode_id"])
        entry = entry_by_id.get(episode_id)
        if entry is None:
            raise ValueError(f"episode {episode_id} is absent from the verified train manifest")
        task = str(entry["task"])
        if str(sample["lang"]).strip() != task:
            raise ValueError(f"task/instruction mismatch for train episode {episode_id}")
        tensors = [
            dino.dino_transform(image.convert("RGB")).unsqueeze(0).to(device)
            for image in (sample["current_image"], sample["terminal_image"])
        ]
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            current, goal = (dino(tensor).float()[0].cpu().numpy() for tensor in tensors)
        residual = goal - current
        step = int(sample["control_step"])
        if step not in global_sums:
            global_sums[step] = np.zeros_like(residual, dtype=np.float64)
        global_sums[step] += residual
        global_counts[step] += 1
        if task not in task_sums[step]:
            task_sums[step][task] = np.zeros_like(residual, dtype=np.float64)
        task_sums[step][task] += residual
        task_counts[step][task] += 1
        episode_ids.add(episode_id)

    missing_steps = set(control_steps) - set(global_sums)
    if missing_steps:
        raise RuntimeError(f"no training samples for requested control steps {sorted(missing_steps)}")
    if not global_sums:
        raise RuntimeError("train split produced no baseline feature samples")
    ordered_steps = sorted(global_sums)
    global_mean = _equal_weight_step_mean(global_sums, global_counts, ordered_steps)
    task_names = sorted(set().union(*(task_sums[step] for step in ordered_steps)))
    arrays = {
        "global_mean_residual": global_mean.astype(np.float32),
    }
    task_keys = {}
    task_counts_by_name = {task: 0 for task in task_names}
    for step in ordered_steps:
        for task, count in task_counts[step].items():
            task_counts_by_name[task] += count
    for index, task in enumerate(task_names):
        task_steps = [step for step in ordered_steps if task_counts[step].get(task, 0)]
        task_mean = _equal_weight_step_mean(
            {step: task_sums[step][task] for step in task_steps},
            {step: task_counts[step][task] for step in task_steps},
            task_steps,
        )
        key = f"task_mean_residual_{index:02d}"
        task_keys[task] = key
        arrays[key] = task_mean.astype(np.float32)
    metadata = {
        "schema_version": 2,
        "baseline_target": "z_terminal - z_current",
        "dataset_split": "train",
        "split_manifest_path": split_info["manifest_path"],
        "split_manifest_sha256": split_info["manifest_sha256"],
        "success_terminal_manifest_sha256": split_info["success_terminal_manifest_sha256"],
        "dataset_metadata_manifest_sha256": split_info["dataset_metadata_manifest_sha256"],
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "dino_backbone": str(cfg.framework.dino.dino_backbone),
        "dino_teacher_sha256": teacher_sha256,
        "image_size": int(cfg.framework.dino.image_size),
        "precision": "BF16 autocast for DINO on CUDA; FP32 accumulation",
        "control_step": ordered_steps[0] if len(ordered_steps) == 1 else None,
        "control_steps": ordered_steps,
        "control_step_aggregation": "equal_weight_over_per_step_means",
        "sample_stride": int(cfg.framework.runtime.vlm_refresh_interval),
        "sample_count": len(dataset),
        "sample_counts_by_control_step": {
            str(step): global_counts[step] for step in ordered_steps
        },
        "episode_count": len(episode_ids),
        "episode_ids": sorted(episode_ids, key=int),
        "task_counts": dict(sorted(task_counts_by_name.items())),
        "task_counts_by_control_step": {
            str(step): dict(sorted(task_counts[step].items())) for step in ordered_steps
        },
        "task_array_keys": task_keys,
        "git": _git_info(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": sys.version,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata_path = args.output.with_suffix(".json")
    metadata["artifact_sha256"] = sha256_file(args.output)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
