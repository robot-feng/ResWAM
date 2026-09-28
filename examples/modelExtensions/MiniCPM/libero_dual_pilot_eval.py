"""LIBERO-side client for the MiniCPM dual-frequency pilot.

Run this file with the dedicated ``libero`` environment. The policy and GPU
model stay in the ResWAM Python 3.10 process; this client sends camera frames
over a local TCP socket and steps one matching LIBERO task.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import logging
import math
import pathlib
import socket
import time

import numpy as np
from PIL import Image

PILOT_EXECUTION_HORIZON = 1


def _select_executed_action(normalized_actions: np.ndarray) -> np.ndarray:
    """Select the evaluator's one committed action from a predicted chunk."""
    chunk = np.asarray(normalized_actions, dtype=np.float32)
    if chunk.ndim == 1:
        return chunk
    if chunk.ndim != 2:
        raise ValueError(f"expected an action or action chunk; got shape {chunk.shape}")
    if chunk.shape[0] < PILOT_EXECUTION_HORIZON:
        raise ValueError(
            f"policy predicted {chunk.shape[0]} actions, but the evaluator "
            f"needs {PILOT_EXECUTION_HORIZON} to execute"
        )
    return chunk[:PILOT_EXECUTION_HORIZON][0]


def _request(sock, payload):
    sock.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
    chunks = bytearray()
    while b"\n" not in chunks:
        block = sock.recv(65536)
        if not block:
            raise ConnectionError("policy socket closed before returning a result")
        chunks.extend(block)
    line, _, rest = chunks.partition(b"\n")
    if rest:
        raise RuntimeError("unexpected extra bytes from policy server")
    response = json.loads(line)
    if not response.get("ok", False):
        raise RuntimeError(f"policy server error: {response.get('error')}")
    return response


def _jpeg_b64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


def _unnormalize_action(action, stats):
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if action.size < 7:
        raise ValueError(f"expected 7 action dimensions; got {action.shape}")
    action = np.clip(action, -1.0, 1.0)
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    action[:6] = 0.5 * (action[:6] + 1.0) * (high[:6] - low[:6]) + low[:6]
    action[6] = float(action[6] >= 0.5)
    # LIBERO's gripper convention is -1=close, +1=open; the dataset label is
    # close=1/open=0, matching the existing LIBERO evaluation adapters.
    action[6] = 1.0 - 2.0 * action[6]
    return action


def _find_task(task_suite, requested_language):
    exact = []
    prefix = []
    needle = requested_language.strip().lower()
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        language = task.language.strip().lower()
        if language == needle:
            exact.append((task_id, task))
        elif language.startswith(needle):
            prefix.append((task_id, task))
    candidates = exact or prefix
    if not candidates:
        raise ValueError(f"No LIBERO task in this suite matches {requested_language!r}")
    # Prefer the unmodified base task when the suite has many generated levels.
    candidates.sort(key=lambda pair: (len(pair[1].language), pair[0]))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task", default="put the bowl on the plate")
    parser.add_argument("--stats", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--max-control-steps", type=int, default=56)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # LIBERO init-state files contain NumPy objects and need the trusted legacy
    # torch.load mode under PyTorch 2.6+.
    import torch

    original_load = torch.load

    def torch_load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = torch_load_compat
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.suite]()

    task_id, task = _find_task(suite, args.task)
    init_states = suite.get_task_init_states(task_id)
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    env.reset()
    obs = env.set_init_state(init_states[0])

    stats_doc = json.loads(pathlib.Path(args.stats).read_text())
    action_stats = stats_doc.get("statistics", stats_doc).get("action", {})
    if "min" not in action_stats or "max" not in action_stats:
        raise ValueError(f"action min/max missing from {args.stats}")

    out_dir = pathlib.Path(args.result).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result_stem = pathlib.Path(args.result).stem
    model_stem = result_stem.removesuffix("_eval")
    video_path = out_dir / f"{model_stem}_rollout.mp4"
    frames = []
    step_latencies = []
    done = False
    control_steps = 0
    final_server_stats = None
    async_step_trace = []

    with socket.create_connection((args.host, args.port), timeout=300) as sock:
        sock.settimeout(300)
        _request(sock, {"type": "reset", "instruction": args.task})
        for _ in range(args.settle_steps):
            obs, _, done, _ = env.step([0.0] * 6 + [-1.0])

        for control_step in range(args.max_control_steps):
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            frames.append(image)
            started = time.perf_counter()
            response = _request(
                sock,
                {
                    "type": "act",
                    "instruction": args.task,
                    "control_step": control_step,
                    "images_jpeg": [_jpeg_b64(image), _jpeg_b64(wrist)],
                },
            )
            step_latencies.append(time.perf_counter() - started)
            final_server_stats = response.get("async_stats")
            if final_server_stats is not None:
                async_step_trace.append(
                    {
                        "control_step": control_step,
                        "cached_vlm_step": final_server_stats.get("latest_vlm_step"),
                        "vlm_age_steps": final_server_stats.get("vlm_age_steps"),
                        "vlm_submitted": final_server_stats.get("vlm_submitted"),
                        "vlm_completed": final_server_stats.get("vlm_completed"),
                        "queued_refreshes": final_server_stats.get("queued_refreshes"),
                        "latest_vlm_activation_event": final_server_stats.get(
                            "latest_vlm_activation_event"
                        ),
                        "policy_roundtrip_seconds": step_latencies[-1],
                    }
                )
            action_norm = _select_executed_action(response["normalized_actions"])
            action = _unnormalize_action(action_norm, action_stats)
            obs, _, done, _ = env.step(action.tolist())
            control_steps += 1
            if done:
                break

    env.close()
    result = {
        "suite": args.suite,
        "task_id": task_id,
        "task_language": task.language,
        "requested_instruction": args.task,
        "seed": args.seed,
        "success": bool(done),
        "control_steps": control_steps,
        "max_control_steps": args.max_control_steps,
        "execution_horizon": PILOT_EXECUTION_HORIZON,
        "execution_policy": "apply the first predicted action, then request a new chunk next control step",
        "async_step_trace": async_step_trace,
        "activation_events": (
            final_server_stats.get("vlm_activation_events", [])
            if final_server_stats is not None
            else []
        ),
        "mean_policy_roundtrip_seconds": float(np.mean(step_latencies)) if step_latencies else 0.0,
        "p95_policy_roundtrip_seconds": float(np.percentile(step_latencies, 95)) if step_latencies else 0.0,
        "async_stats": final_server_stats,
        "video": str(video_path),
    }
    pathlib.Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(str(video_path), frames, fps=10)
    except Exception as exc:
        result["video_error"] = str(exc)
        pathlib.Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
