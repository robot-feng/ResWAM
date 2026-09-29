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
    parser.add_argument("--control-step", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

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
        control_steps=[args.control_step],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = get_dino_model(str(cfg.framework.dino.dino_backbone)).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    teacher_sha256 = _module_sha256(dino)

    global_sum = None
    task_sums: dict[str, np.ndarray] = defaultdict(lambda: None)
    task_counts: dict[str, int] = defaultdict(int)
    episode_ids = []
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
        if global_sum is None:
            global_sum = np.zeros_like(residual, dtype=np.float64)
        global_sum += residual
        if task_sums[task] is None:
            task_sums[task] = np.zeros_like(residual, dtype=np.float64)
        task_sums[task] += residual
        task_counts[task] += 1
        episode_ids.append(episode_id)

    if global_sum is None:
        raise RuntimeError("train split produced no baseline feature samples")
    task_names = sorted(task_sums)
    arrays = {
        "global_mean_residual": (global_sum / len(dataset)).astype(np.float32),
    }
    task_keys = {}
    for index, task in enumerate(task_names):
        key = f"task_mean_residual_{index:02d}"
        task_keys[task] = key
        arrays[key] = (task_sums[task] / task_counts[task]).astype(np.float32)
    metadata = {
        "schema_version": 1,
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
        "control_step": args.control_step,
        "sample_stride": int(cfg.framework.runtime.vlm_refresh_interval),
        "sample_count": len(dataset),
        "episode_ids": sorted(episode_ids, key=int),
        "task_counts": dict(sorted(task_counts.items())),
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
