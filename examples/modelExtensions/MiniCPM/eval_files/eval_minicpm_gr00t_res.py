#!/usr/bin/env python3
"""Evaluate residual-vs-zero baseline and profile configured history lengths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.minicpm_res_lerobot import MiniCPMResidualLeRobotDataset
from starVLA.dataloader.minicpm_res_splits import (
    SPLIT_NAMES,
    load_residual_episode_split,
    sha256_file,
)
from starVLA.model.framework.VLM4A.MiniCPMGR00TRes import MiniCPMGR00TRes
from starVLA.model.framework.VLM4A.minicpm_video_history import gather_token_hidden_states
from starVLA.model.framework.VLM4A.minicpm_res_metrics import goal_space_metrics


def _git_info() -> dict[str, str | bool | None]:
    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _load_cfg(path: Path):
    cfg = OmegaConf.load(path)
    inherited = cfg.get("defaults")
    if inherited:
        if not OmegaConf.is_list(inherited) or len(inherited) != 1 or not isinstance(inherited[0], str):
            raise ValueError("only one string config parent is supported in defaults")
        parent_path = (path.parent / inherited[0]).with_suffix(".yaml")
        del cfg["defaults"]
        cfg = OmegaConf.merge(OmegaConf.load(parent_path), cfg)
    return cfg


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument(
        "--episode-ids", default=None, help="optional comma-separated episode IDs for a held-out split"
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=SPLIT_NAMES, default=None)
    parser.add_argument("--control-steps", default=None, help="optional comma-separated control-step indices")
    parser.add_argument("--profile-history-lengths", default="1,8,32,all")
    parser.add_argument(
        "--baseline-means",
        type=Path,
        default=None,
        help="optional train-only global/task mean residual NPZ for constant-predictor baselines",
    )
    parser.add_argument(
        "--mask-history-frame",
        choices=("none", "oldest", "middle", "most_recent_previous"),
        default="none",
        help="Replace a selected past frame with a neutral gray frame while preserving its time/position.",
    )
    parser.add_argument("--output", type=Path, default=Path("reswam_eval_profile.json"))
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    dataset_cfg = cfg.datasets.residual_data
    episode_ids = (
        None if args.episode_ids is None else [item.strip() for item in args.episode_ids.split(",") if item.strip()]
    )
    split_info = None
    if args.split_manifest is not None or args.split is not None:
        if args.split_manifest is None or args.split is None:
            raise ValueError("--split-manifest and --split must be provided together")
        if episode_ids is not None:
            raise ValueError("--episode-ids cannot be combined with a frozen --split-manifest")
        split_info = load_residual_episode_split(
            args.split_manifest,
            split=args.split,
            success_terminal_manifest=dataset_cfg.success_terminal_manifest,
            dataset_path=Path(dataset_cfg.data_root_dir) / dataset_cfg.dataset_name,
        )
        episode_ids = split_info["episode_ids"]
    control_steps = (
        None
        if args.control_steps is None
        else [int(item.strip()) for item in args.control_steps.split(",") if item.strip()]
    )
    dataset = MiniCPMResidualLeRobotDataset(
        data_root_dir=dataset_cfg.data_root_dir,
        dataset_name=dataset_cfg.dataset_name,
        robot_type=dataset_cfg.robot_type,
        success_terminal_manifest=dataset_cfg.success_terminal_manifest,
        assistant_label_manifest=dataset_cfg.get("assistant_label_manifest"),
        camera_key=dataset_cfg.camera_key,
        video_backend=dataset_cfg.video_backend,
        data_cfg={"video_backend": dataset_cfg.video_backend, "include_state": False},
        max_history_frames=dataset_cfg.get("max_history_frames"),
        history_mode=str(cfg.framework.residual_model.history_mode),
        sample_stride=int(cfg.framework.runtime.vlm_refresh_interval),
        episode_ids=episode_ids,
        control_steps=control_steps,
    )
    model = MiniCPMGR00TRes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_reswam_checkpoint(args.checkpoint)
    if (
        args.mask_history_frame != "none"
        and str(model.config.framework.residual_model.history_mode) != "full"
    ):
        raise ValueError("history-frame masking requires residual_model.history_mode='full'")
    torch.set_grad_enabled(False)

    mean_baselines = None
    mean_baseline_metadata = None
    if args.baseline_means is not None:
        with np.load(args.baseline_means, allow_pickle=False) as baseline_artifact:
            mean_baseline_metadata = json.loads(str(baseline_artifact["metadata_json"].item()))
            if split_info is None:
                raise ValueError("train-mean baselines require a verified frozen split manifest")
            if mean_baseline_metadata["split_manifest_sha256"] != split_info["manifest_sha256"]:
                raise ValueError("mean-residual baseline was built from a different split manifest")
            if mean_baseline_metadata["dino_teacher_sha256"] != model.dino_teacher_sha256:
                raise ValueError("mean-residual baseline uses a different DINO teacher")
            if mean_baseline_metadata.get("dataset_split") != "train":
                raise ValueError("global/task mean baselines must be computed from train only")
            expected_tasks = {str(entry["task"]) for entry in split_info["entries"]}
            if set(mean_baseline_metadata["task_array_keys"]) != expected_tasks:
                raise ValueError("mean-residual task baselines do not cover exactly the evaluated split tasks")
            if (
                control_steps is not None
                and int(mean_baseline_metadata["control_step"]) not in control_steps
            ):
                raise ValueError("mean-residual baseline control step is absent from evaluation control steps")
            mean_baselines = {
                "global_mean_residual": torch.from_numpy(
                    baseline_artifact["global_mean_residual"].copy()
                ).unsqueeze(0).to(device),
                "task_mean_residuals": {
                    task: torch.from_numpy(baseline_artifact[key].copy()).unsqueeze(0).to(device)
                    for task, key in mean_baseline_metadata["task_array_keys"].items()
                },
            }

    requested = [part.strip() for part in args.profile_history_lengths.split(",") if part.strip()]
    results = {
        "framework": "MiniCPMGR00TRes",
        "goal_target": "successful_terminal",
        "prediction_target": model.prediction_target,
        "dino_teacher_sha256": model.dino_teacher_sha256,
        "history_mode": str(model.config.framework.residual_model.history_mode),
        "mask_history_frame": args.mask_history_frame,
        "vlm_refresh_interval": model.vlm_refresh_interval,
        "action_horizon": "owned by downstream action policy; not consumed here",
        "execution_horizon": "owned by downstream evaluator; not consumed here",
        "alignment_mode": "synchronous_terminal_representation_evaluation",
        "data_split": (
            {
                "split": split_info["split"],
                "manifest_path": split_info["manifest_path"],
                "manifest_sha256": split_info["manifest_sha256"],
                "dataset_metadata_manifest_sha256": split_info[
                    "dataset_metadata_manifest_sha256"
                ],
                "success_terminal_manifest_sha256": split_info[
                    "success_terminal_manifest_sha256"
                ],
                "episode_ids": split_info["episode_ids"],
            }
            if split_info is not None
            else None
        ),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(
            args.checkpoint if args.checkpoint.is_file() else args.checkpoint / "trainable_state.pt"
        ),
        "baseline_means_path": None if args.baseline_means is None else str(args.baseline_means.resolve()),
        "baseline_means_sha256": None if args.baseline_means is None else sha256_file(args.baseline_means),
        "baseline_means_metadata": mean_baseline_metadata,
        "git": _git_info(),
        "precision": "BF16 autocast for MiniCPM and DINO on CUDA; FP32 head and metrics",
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "evaluation_args": {
            "max_samples": args.max_samples,
            "control_steps": control_steps,
            "profile_history_lengths": requested,
        },
        "device": str(device),
        "history_profiles": [],
    }
    for profile in requested:
        dataset.max_history_frames = None if profile == "all" else int(profile)
        rows = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for index in range(min(args.max_samples, len(dataset))):
            sample = dataset[index]
            history, instruction, episode_id = model._history_from_example(sample)
            masked_history_index = None
            if args.mask_history_frame != "none":
                if len(history) < 2:
                    raise ValueError("history masking requires at least one past frame and one current frame")
                if args.mask_history_frame == "oldest":
                    masked_history_index = 0
                elif args.mask_history_frame == "middle":
                    masked_history_index = (len(history) - 1) // 2
                else:
                    masked_history_index = len(history) - 2
                if masked_history_index == len(history) - 1:
                    raise ValueError("the current frame cannot be selected for history masking")
                neutral = Image.new("RGB", history[masked_history_index].image.size, (127, 127, 127))
                history[masked_history_index] = replace(
                    history[masked_history_index], image=neutral
                )
            t0 = time.perf_counter()
            inputs = model.history_adapter.tokenize_history_query(
                instruction,
                episode_id,
                history,
                history_mode=str(model.config.framework.residual_model.history_mode),
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
            raw_prediction = model.residual_head(query.float())
            _sync(device)
            query_head_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            current = model.encode_dino(sample["current_image"])
            target_goal = model.encode_dino(sample["terminal_image"])
            target_residual = target_goal - current
            predicted_residual = (
                raw_prediction.float()
                if model.prediction_target == "residual"
                else raw_prediction.float() - current
            )
            _sync(device)
            dino_teacher_seconds = time.perf_counter() - t0

            predictors = {
                "model": predicted_residual,
                "zero_residual": torch.zeros_like(target_residual),
            }
            if mean_baselines is not None:
                predictors["global_mean_residual"] = mean_baselines["global_mean_residual"]
                task_residual = mean_baselines["task_mean_residuals"].get(instruction)
                if task_residual is not None:
                    predictors["task_mean_residual"] = task_residual
            metrics = {
                name: goal_space_metrics(prediction, target_residual, current)
                for name, prediction in predictors.items()
            }

            rows.append(
                {
                    "episode_id": sample["episode_id"],
                    "control_step": sample["control_step"],
                    "history_frames": len(history),
                    "context_tokens": context_tokens,
                    "masked_history_index": masked_history_index,
                    "predictor_metrics": metrics,
                    "preprocess_seconds": preprocess_seconds,
                    "vlm_seconds": vlm_seconds,
                    "query_and_head_seconds": query_head_seconds,
                    "dino_teacher_seconds": dino_teacher_seconds,
                }
            )
        if not rows:
            raise RuntimeError(f"no samples evaluated for history profile {profile!r}")
        predictor_names = set(rows[0]["predictor_metrics"])
        if any(set(row["predictor_metrics"]) != predictor_names for row in rows):
            raise RuntimeError("predictor metric keys changed between evaluation samples")
        metric_names = tuple(rows[0]["predictor_metrics"]["model"])
        mean_metrics_by_predictor = {
            predictor: {
                metric: sum(row["predictor_metrics"][predictor][metric] for row in rows) / len(rows)
                for metric in metric_names
            }
            for predictor in rows[0]["predictor_metrics"]
        }
        results["history_profiles"].append(
            {
                "requested_frames": profile,
                "samples": len(rows),
                "mean_history_frames": sum(row["history_frames"] for row in rows) / len(rows),
                "mean_context_tokens": sum(row["context_tokens"] for row in rows) / len(rows),
                "mean_goal_mse": mean_metrics_by_predictor["model"]["goal_mse"],
                "mean_zero_residual_goal_mse": mean_metrics_by_predictor["zero_residual"]["goal_mse"],
                "mean_metrics_by_predictor": mean_metrics_by_predictor,
                "mean_preprocess_seconds": sum(row["preprocess_seconds"] for row in rows) / len(rows),
                "mean_vlm_seconds": sum(row["vlm_seconds"] for row in rows) / len(rows),
                "mean_query_and_head_seconds": sum(row["query_and_head_seconds"] for row in rows) / len(rows),
                "mean_dino_teacher_seconds": sum(row["dino_teacher_seconds"] for row in rows) / len(rows),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "rows": rows,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
