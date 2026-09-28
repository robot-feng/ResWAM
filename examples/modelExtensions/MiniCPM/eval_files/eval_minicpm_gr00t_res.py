#!/usr/bin/env python3
"""Evaluate residual-vs-zero baseline and profile configured history lengths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.minicpm_res_lerobot import MiniCPMResidualLeRobotDataset
from starVLA.model.framework.VLM4A.MiniCPMGR00TRes import MiniCPMGR00TRes
from starVLA.model.framework.VLM4A.minicpm_video_history import gather_token_hidden_states


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


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--profile-history-lengths", default="1,8,32,all")
    parser.add_argument("--output", type=Path, default=Path("reswam_eval_profile.json"))
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    dataset_cfg = cfg.datasets.residual_data
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
        sample_stride=int(cfg.framework.runtime.execution_horizon),
    )
    model = MiniCPMGR00TRes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_reswam_checkpoint(args.checkpoint)
    torch.set_grad_enabled(False)

    requested = [part.strip() for part in args.profile_history_lengths.split(",") if part.strip()]
    results = {
        "framework": "MiniCPMGR00TRes",
        "goal_target": "successful_terminal",
        "prediction_target": model.prediction_target,
        "execution_horizon": model.execution_horizon,
        "action_horizon": "owned by downstream action policy; not consumed here",
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
            prediction = model.residual_head(query.float())
            _sync(device)
            query_head_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            current = model.encode_dino(sample["current_image"])
            target_goal = model.encode_dino(sample["terminal_image"])
            target_residual = target_goal - current
            if model.prediction_target == "absolute_goal":
                prediction = prediction - current
            _sync(device)
            dino_teacher_seconds = time.perf_counter() - t0

            rows.append(
                {
                    "episode_id": sample["episode_id"],
                    "control_step": sample["control_step"],
                    "history_frames": len(history),
                    "context_tokens": context_tokens,
                    "residual_mse": float(torch.mean((prediction.float() - target_residual.float()) ** 2)),
                    "zero_residual_mse": float(torch.mean(target_residual.float() ** 2)),
                    "preprocess_seconds": preprocess_seconds,
                    "vlm_seconds": vlm_seconds,
                    "query_and_head_seconds": query_head_seconds,
                    "dino_teacher_seconds": dino_teacher_seconds,
                }
            )
        if not rows:
            raise RuntimeError(f"no samples evaluated for history profile {profile!r}")
        results["history_profiles"].append(
            {
                "requested_frames": profile,
                "samples": len(rows),
                "mean_history_frames": sum(row["history_frames"] for row in rows) / len(rows),
                "mean_context_tokens": sum(row["context_tokens"] for row in rows) / len(rows),
                "mean_residual_mse": sum(row["residual_mse"] for row in rows) / len(rows),
                "mean_zero_residual_mse": sum(row["zero_residual_mse"] for row in rows) / len(rows),
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
