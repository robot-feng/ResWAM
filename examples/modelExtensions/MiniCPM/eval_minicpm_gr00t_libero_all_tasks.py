"""Evaluate a trained MiniCPM-GR00T checkpoint on every LIBERO Goal task.

The policy runs in the ResWAM environment and the simulator client runs in the
dedicated LIBERO environment, matching ``libero_dual_pilot.py``'s transport.
For DualAsy, simulator evaluation always uses the real wall-clock refresh path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import torch
from omegaconf import OmegaConf

from libero_dual_pilot import DEFAULT_DATA_ROOT, _serve_and_simulate
from starVLA.model.framework.base_framework import build_framework


ROOT = pathlib.Path(__file__).resolve().parents[3]
TASKS_PATH = DEFAULT_DATA_ROOT / "libero_goal_no_noops_1.0.0_lerobot/meta/tasks.jsonl"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tasks(path: pathlib.Path):
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tasks.sort(key=lambda row: int(row["task_index"]))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--task-indices", nargs="+", type=int, default=None)
    parser.add_argument("--init-state-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--max-control-steps", type=int, default=300)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.max_control_steps < 1:
        raise ValueError("--max-control-steps must be >= 1")
    if not args.init_state_indices or min(args.init_state_indices) < 0:
        raise ValueError("--init-state-indices must contain nonnegative indices")

    cfg = OmegaConf.load(config_path)
    framework_name = str(cfg.framework.name)
    if framework_name not in ("MiniCPMGR00TDual", "MiniCPMGR00TDualAsy"):
        raise ValueError(f"unsupported framework for this evaluator: {framework_name}")
    if framework_name == "MiniCPMGR00TDualAsy":
        # Training used a calibrated fixed-step replay; deployment evaluation
        # must test nonblocking refreshes against actual wall-clock latency.
        cfg.framework.async_alignment = {
            "mode": "wall_clock",
            "fixed_latency_steps": None,
            "trace_path": None,
            "training_trace_path": None,
            "trace_max_control_step": None,
        }

    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = checkpoint_path.parent / f"libero_goal_eval_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    tasks = _load_tasks(TASKS_PATH)
    task_indices = (
        list(range(len(tasks))) if args.task_indices is None else args.task_indices
    )
    invalid_tasks = [index for index in task_indices if not 0 <= index < len(tasks)]
    if invalid_tasks:
        raise IndexError(f"task indices outside [0, {len(tasks)}): {invalid_tasks}")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be >= 1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[libero-eval] framework={framework_name} device={device}")
    print(f"[libero-eval] checkpoint={checkpoint_path}")
    print(f"[libero-eval] output={output_dir}")

    model = build_framework(cfg).to(device).eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    provenance = {
        "framework": framework_name,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "runtime_async_alignment": (
            "wall_clock" if framework_name == "MiniCPMGR00TDualAsy" else None
        ),
        "suite": "libero_goal",
        "max_control_steps": args.max_control_steps,
        "execution_horizon": args.execution_horizon,
        "init_state_indices": args.init_state_indices,
        "seed": args.seed,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    records = []
    started = time.perf_counter()
    try:
        for task_index in task_indices:
            task = tasks[task_index]
            instruction = str(task["task"])
            for init_index in args.init_state_indices:
                result_filename = (
                    f"{framework_name}_task{task_index:02d}_init{init_index:02d}.json"
                )
                run_seed = args.seed + task_index * 1000 + init_index
                print(
                    f"[libero-eval] task {task_index + 1}/{len(tasks)} "
                    f"init={init_index}: {instruction}"
                )
                try:
                    result = _serve_and_simulate(
                        model,
                        framework_name,
                        output_dir,
                        instruction,
                        args.max_control_steps,
                        run_seed,
                        args.execution_horizon,
                        init_state_index=init_index,
                        settle_steps=args.settle_steps,
                        result_filename=result_filename,
                        close_async_worker=False,
                    )
                    record = {
                        "task_index": task_index,
                        "task": instruction,
                        "init_state_index": init_index,
                        "seed": run_seed,
                        "success": bool(result["success"]),
                        "control_steps": int(result["control_steps"]),
                        "mean_policy_roundtrip_seconds": float(
                            result.get("mean_policy_roundtrip_seconds", 0.0)
                        ),
                        "p95_policy_roundtrip_seconds": float(
                            result.get("p95_policy_roundtrip_seconds", 0.0)
                        ),
                        "async_stats": result.get("async_stats"),
                        "result_path": str(output_dir / result_filename),
                        "video_path": result.get("video"),
                    }
                except Exception as exc:  # preserve other task results on an isolated simulator error
                    record = {
                        "task_index": task_index,
                        "task": instruction,
                        "init_state_index": init_index,
                        "seed": run_seed,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[libero-eval] ERROR: {record['error']}")
                records.append(record)
                summary = {
                    **provenance,
                    "completed_episodes": len(records),
                    "successes": sum(bool(item.get("success")) for item in records),
                    "success_rate": sum(bool(item.get("success")) for item in records)
                    / len(records),
                    "elapsed_seconds": time.perf_counter() - started,
                    "episodes": records,
                }
                (output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
    finally:
        if hasattr(model, "close_async_worker"):
            model.close_async_worker()

    successes = sum(bool(item.get("success")) for item in records)
    aggregate = {
        **provenance,
        "completed_episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "episodes": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: aggregate[key] for key in (
        "framework", "completed_episodes", "successes", "success_rate", "elapsed_seconds"
    )}, indent=2))
    print(f"[libero-eval] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
