"""Small same-trajectory LIBERO smoke pipeline, stage 1 by default.

The default run trains and evaluates only ``MiniCPMGR00TDual``. Later
baselines can be included explicitly with ``--models``; the dual-asynchronous
config and anchor-frame alignment are loaded only when that model is selected.
The smoke uses episode 0 from ``libero_goal`` for a few optimizer steps and can
evaluate one matching LIBERO task. The VLM and DINO encoder are frozen; the
action head and DINO projection are trained.

The LIBERO simulator runs in the separate Python 3.12 ``libero`` environment.
The script serves each live policy over localhost TCP, so no checkpoint copy or
model installation is needed in that environment.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import socket
import socketserver
import subprocess
import sys
import threading
import time
import io

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.minicpm_dual_asy_alignment import (
    ALIGNMENT_MODES,
    load_trace_activation_steps,
    load_trace_max_control_step,
    source_step_at,
)

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = pathlib.Path("/data/tzq/datasets/starVLA/Datasets/libero_10hz")
DEFAULT_CONFIG = ROOT / "examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_dual_libero.yaml"
DEFAULT_ASY_CONFIG = ROOT / "examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_dual_asy_libero.yaml"
LIBERO_ROOT = pathlib.Path("/home/taizun/lcy/FastWAM/third_party/LIBERO-plus")
LIBERO_PYTHON = pathlib.Path("/data/miniconda3/envs/libero/bin/python")
MODEL_IDS = ("MiniCPMGR00T", "MiniCPMGR00TDual", "MiniCPMGR00TDualAsy")
TRAIN_FRAME_IDS = (1, 2, 9, 10)
DATASET_ID = "libero_goal_no_noops_1.0.0_lerobot"


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _git_provenance():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty_worktree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_worktree": None}


def _dataset_metadata_manifest(dataset_path):
    dataset_path = pathlib.Path(dataset_path)
    meta_path = dataset_path / "meta"
    candidates = (
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "modality.json",
        "downsample.json",
        "stats_gr00t.json",
    )
    files = {
        name: _sha256_file(meta_path / name)
        for name in candidates
        if (meta_path / name).is_file()
    }
    return {
        "dataset_path": str(dataset_path),
        "metadata_file_sha256": files,
        "manifest_sha256": _canonical_json_hash(files) if files else None,
    }


def _dependency_versions():
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("transformers", "torchvision", "timm", "modelscope", "omegaconf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_provenance():
    if not torch.cuda.is_available():
        return {"cuda_available": False, "devices": []}
    return {
        "cuda_available": True,
        "cuda_runtime": torch.version.cuda,
        "devices": [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ],
    }


def _module_parameter_dtype(module):
    try:
        return str(next(module.parameters()).dtype)
    except (AttributeError, StopIteration):
        return None


def _save_trainable_checkpoint(model, framework_name, output_dir, seed, config_sha256):
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_path = pathlib.Path(output_dir) / f"{framework_name}_trainable.pt"
    torch.save(
        {
            "framework": framework_name,
            "seed": int(seed),
            "resolved_config_sha256": config_sha256,
            "trainable_state_dict": trainable_state,
        },
        checkpoint_path,
    )
    return {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "trainable_parameter_tensors": len(trainable_state),
        "trainable_parameter_count": sum(tensor.numel() for tensor in trainable_state.values()),
        "scope": "trainable parameters only; frozen base models are referenced in the resolved config",
    }


def _resolve_async_refresh_interval(models, requested_interval, async_config_path):
    """Load async-only settings only when the stage-2 model is selected."""
    run_async = "MiniCPMGR00TDualAsy" in models
    if requested_interval is not None and not run_async:
        raise ValueError("--vlm-refresh-interval is only valid when MiniCPMGR00TDualAsy is selected")
    if not run_async:
        return None

    async_cfg = OmegaConf.load(async_config_path)
    interval = requested_interval
    if interval is None:
        interval = async_cfg.framework.get("vlm_refresh_interval")
    if interval is None:
        interval = 8
    interval = int(interval)
    if interval < 1:
        raise ValueError("framework.vlm_refresh_interval must be >= 1")
    return interval


def _latest_completed_vlm_anchor_step(control_step, refresh_interval, latency_steps):
    """Backward-compatible helper for the fixed-step-delay study mode."""
    return source_step_at(
        control_step,
        mode="fixed_step_delay",
        refresh_interval=refresh_interval,
        fixed_latency_steps=latency_steps,
    )


def _resolve_async_alignment(
    models,
    requested_mode,
    requested_latency_steps,
    requested_trace,
    requested_training_trace,
    async_config_path,
):
    """Resolve distinct runtime and offline-training alignment policies.

    A wall-clock runtime cannot be reconstructed from a scalar offline delay.
    It trains against an explicitly recorded activation trace instead.
    """
    run_async = "MiniCPMGR00TDualAsy" in models
    options = (
        requested_mode,
        requested_latency_steps,
        requested_trace,
        requested_training_trace,
    )
    if not run_async:
        if any(value is not None for value in options):
            raise ValueError(
                "alignment options are only valid when MiniCPMGR00TDualAsy is selected"
            )
        return None

    async_cfg = OmegaConf.load(async_config_path)
    framework_cfg = async_cfg.get("framework") or {}
    configured_alignment = framework_cfg.get("async_alignment") or {}
    mode = requested_mode or configured_alignment.get("mode")
    if mode is None:
        raise ValueError(
            "select --alignment-mode explicitly for MiniCPMGR00TDualAsy; "
            "there is no default VLM latency"
        )
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"--alignment-mode must be one of {ALIGNMENT_MODES}")

    fixed_latency = requested_latency_steps
    if fixed_latency is None:
        fixed_latency = configured_alignment.get("fixed_latency_steps")
    if mode == "fixed_step_delay":
        if fixed_latency is None or int(fixed_latency) < 0:
            raise ValueError(
                "fixed_step_delay requires --fixed-latency-steps >= 0"
            )
        fixed_latency = int(fixed_latency)
    elif fixed_latency is not None:
        raise ValueError(
            "--fixed-latency-steps is only valid with fixed_step_delay; "
            "wall_clock delivery delay is measured at runtime"
        )

    runtime_mode = mode
    trace_path = (
        requested_trace
        or configured_alignment.get("trace_path")
    )
    training_trace = requested_training_trace or trace_path
    if mode == "wall_clock":
        if not training_trace:
            raise ValueError(
                "wall_clock training alignment requires a prior activation trace; "
                "pass --training-alignment-trace"
            )
        training_mode = "trace_replay"
        trace_path = training_trace
    elif mode == "trace_replay":
        if not trace_path:
            raise ValueError("trace_replay requires --alignment-trace")
        training_mode = "trace_replay"
    else:
        training_mode = mode

    trace_events = None
    if training_mode == "trace_replay":
        trace_events = load_trace_activation_steps(trace_path)
    runtime_trace_path = (
        str(pathlib.Path(trace_path).resolve())
        if runtime_mode == "trace_replay" and trace_path
        else None
    )
    training_trace_path = (
        str(pathlib.Path(trace_path).resolve())
        if training_mode == "trace_replay" and trace_path
        else None
    )
    runtime_trace_max_control_step = (
        load_trace_max_control_step(runtime_trace_path)
        if runtime_mode == "trace_replay"
        else None
    )
    return {
        "runtime_mode": runtime_mode,
        "training_mode": training_mode,
        "fixed_latency_steps": fixed_latency,
        "runtime_trace_path": runtime_trace_path,
        "training_trace_path": training_trace_path,
        "runtime_trace_max_control_step": runtime_trace_max_control_step,
        "trace_activation_steps": trace_events,
    }


def _json_send(handler, payload):
    handler.wfile.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    handler.wfile.flush()


class _PolicyServer(socketserver.StreamRequestHandler):
    def handle(self):
        policy = self.server.policy
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line)
                kind = request.get("type")
                if kind == "reset":
                    if hasattr(policy, "reset_async_cache"):
                        policy.reset_async_cache()
                    _json_send(self, {"ok": True})
                    continue
                if kind == "stats":
                    if hasattr(policy, "wait_for_async_refreshes"):
                        policy.wait_for_async_refreshes(timeout=120)
                    stats = policy.async_stats() if hasattr(policy, "async_stats") else None
                    _json_send(self, {"ok": True, "async_stats": stats})
                    continue
                if kind != "act":
                    raise ValueError(f"unknown request type {kind!r}")

                images = []
                for encoded in request["images_jpeg"]:
                    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                    images.append(image)
                example = {"image": images, "lang": request["instruction"]}
                step = int(request["control_step"])
                if hasattr(policy, "reset_async_cache"):
                    output = policy.predict_action([example], control_step=step)
                else:
                    output = policy.predict_action([example])
                _json_send(
                    self,
                    {
                        "ok": True,
                        "normalized_actions": np.asarray(output["normalized_actions"])[0].tolist(),
                        "async_stats": output.get("async_stats"),
                    },
                )
            except Exception as exc:
                _json_send(self, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class _ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, policy):
        super().__init__(address, _PolicyServer)
        self.policy = policy


def _build_examples(
    dataset,
    frame_ids,
    vlm_refresh_interval,
    asynchronous,
    alignment_mode=None,
    fixed_latency_steps=None,
    trace_activation_steps=None,
):
    examples = []
    for frame_id in frame_ids:
        example = copy.deepcopy(dataset[frame_id])
        if asynchronous:
            if alignment_mode is None:
                raise ValueError("asynchronous training requires an explicit alignment_mode")
            if alignment_mode == "fixed_step_delay" and fixed_latency_steps is None:
                raise ValueError(
                    "fixed_step_delay training requires explicit fixed_latency_steps"
                )
            anchor_id = source_step_at(
                frame_id,
                mode=alignment_mode,
                refresh_interval=vlm_refresh_interval,
                fixed_latency_steps=(
                    fixed_latency_steps if alignment_mode == "fixed_step_delay" else None
                ),
                trace_activation_steps=trace_activation_steps,
            )
            example["vlm_image"] = copy.deepcopy(dataset[anchor_id]["image"])
            example["vlm_anchor_frame"] = anchor_id
        examples.append(example)
    return examples


def _freeze_backbones(model, include_dino):
    for parameter in model.qwen_vl_interface.parameters():
        parameter.requires_grad_(False)
    model.qwen_vl_interface.eval()
    if include_dino:
        for parameter in model.dino_encoder.parameters():
            parameter.requires_grad_(False)
        model.dino_encoder.eval()


def _new_model(framework_name, cfg, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if framework_name == "MiniCPMGR00T":
        from starVLA.model.framework.VLM4A.MiniCPMGR00T import MiniCPM_GR00T

        model = MiniCPM_GR00T(cfg)
    elif framework_name == "MiniCPMGR00TDual":
        from starVLA.model.framework.VLM4A.MiniCPMGR00TDual import MiniCPMGR00TDual

        model = MiniCPMGR00TDual(cfg)
    else:
        from starVLA.model.framework.VLM4A.MiniCPMGR00TDualAsy import MiniCPMGR00TDualAsy

        model = MiniCPMGR00TDualAsy(cfg)
    model.to(device)
    return model


def _run_training(model, framework_name, examples, heldout, learning_rate):
    dual = framework_name != "MiniCPMGR00T"
    _freeze_backbones(model, include_dino=dual)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"{framework_name} has no trainable action parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-8)
    model.train()
    model.qwen_vl_interface.eval()
    if dual:
        model.dino_encoder.eval()

    records = []
    started = time.perf_counter()
    for update, example in enumerate(examples, start=1):
        optimizer.zero_grad(set_to_none=True)
        output = model([example])
        loss = output["action_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{framework_name} produced non-finite loss at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"{framework_name} produced non-finite gradients")
        optimizer.step()
        records.append({"step": update, "action_loss": float(loss.detach()), "grad_norm": float(grad_norm)})

    model.eval()
    model.qwen_vl_interface.eval()
    if dual:
        model.dino_encoder.eval()
    with torch.inference_mode():
        heldout_loss = float(model([heldout])["action_loss"].detach())
    return {
        "train_steps": len(records),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "train_losses": records,
        "heldout_action_loss": heldout_loss,
        "training_seconds": time.perf_counter() - started,
    }


def _serve_and_simulate(model, framework_name, output_dir, instruction, max_steps, seed):
    if hasattr(model, "reset_async_cache"):
        model.reset_async_cache()
    server = _ThreadedTCPServer(("127.0.0.1", 0), model)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    result_path = output_dir / f"{framework_name}_eval.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(LIBERO_ROOT),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            # robosuite reads CUDA_VISIBLE_DEVICES to choose its EGL device;
            # an empty value makes that library parse an empty device id.
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        str(LIBERO_PYTHON),
        str(ROOT / "examples/modelExtensions/MiniCPM/libero_dual_pilot_eval.py"),
        "--port",
        str(port),
        "--suite",
        "libero_goal",
        "--task",
        instruction,
        "--stats",
        str(DEFAULT_DATA_ROOT / "libero_goal_no_noops_1.0.0_lerobot/meta/stats_gr00t.json"),
        "--result",
        str(result_path),
        "--max-control-steps",
        str(max_steps),
        "--seed",
        str(seed),
    ]
    try:
        completed = subprocess.run(command, env=env, check=False, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                f"LIBERO evaluator exited with code {completed.returncode}; "
                f"stderr tail:\n{completed.stderr[-4000:]}"
            )
        if hasattr(model, "wait_for_async_refreshes"):
            model.wait_for_async_refreshes(timeout=120)
        if hasattr(model, "async_stats"):
            stats = model.async_stats()
            result = json.loads(result_path.read_text())
            result["async_stats"] = stats
            result["activation_events"] = stats.get("vlm_activation_events", [])
            result["refresh_events"] = stats.get("vlm_refresh_events", [])
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        return json.loads(result_path.read_text())
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        if hasattr(model, "close_async_worker"):
            model.close_async_worker()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--async-config", type=pathlib.Path, default=DEFAULT_ASY_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--max-control-steps", type=int, default=56)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_IDS,
        default=("MiniCPMGR00TDual",),
        help="model variants to run (defaults to the stage-1 dual baseline)",
    )
    parser.add_argument(
        "--vlm-refresh-interval",
        type=int,
        default=None,
        help="control steps between upper VLM refreshes (defaults to 8)",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=ALIGNMENT_MODES,
        default=None,
        help="shared training/runtime alignment mode; wall_clock training uses an explicit replay trace",
    )
    parser.add_argument(
        "--fixed-latency-steps",
        type=int,
        default=None,
        help="controlled worker delay for fixed_step_delay; never assumed for wall_clock",
    )
    parser.add_argument("--alignment-trace", type=pathlib.Path, default=None)
    parser.add_argument("--training-alignment-trace", type=pathlib.Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

    if not args.skip_simulation:
        if not LIBERO_PYTHON.is_file():
            raise FileNotFoundError(f"LIBERO_PYTHON not found: {LIBERO_PYTHON}")
        if not LIBERO_ROOT.is_dir():
            raise FileNotFoundError(f"LIBERO root not found: {LIBERO_ROOT}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / "playground/Checkpoints/libero_minicpm_pilot" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    base_cfg = OmegaConf.load(args.config)
    base_cfg.framework.action_model.action_horizon = int(
        base_cfg.framework.action_model.get("action_horizon", 8)
    )
    action_chunk_length = int(base_cfg.framework.action_model.action_horizon)
    run_async = "MiniCPMGR00TDualAsy" in args.models
    vlm_refresh_interval = _resolve_async_refresh_interval(
        args.models, args.vlm_refresh_interval, args.async_config
    )
    alignment = _resolve_async_alignment(
        args.models,
        args.alignment_mode,
        args.fixed_latency_steps,
        args.alignment_trace,
        args.training_alignment_trace,
        args.async_config,
    )
    if action_chunk_length < 1:
        raise ValueError("framework.action_model.action_horizon must be >= 1")

    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    data_cfg = {"lerobot_version": "v2.0", "video_backend": "torchvision_av"}
    dataset = make_LeRobotSingleDataset(
        args.data_root,
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_franka",
        data_cfg=data_cfg,
    )
    episode0_steps = [i for i, (episode_id, _) in enumerate(dataset.all_steps) if int(episode_id) == 0]
    if len(episode0_steps) < 24 or episode0_steps[:3] != [0, 1, 2]:
        raise RuntimeError(f"could not identify the expected episode 0 window: {episode0_steps[:10]}")
    frame_count = len(episode0_steps)
    max_frame = min(frame_count - action_chunk_length, 47)
    selected = [int(x) for x in TRAIN_FRAME_IDS if x < max_frame]
    if len(selected) < args.train_steps:
        raise ValueError(f"episode 0 has too few valid frames for {args.train_steps} updates")
    selected = selected[: args.train_steps]

    # Materialize only the selected single-trajectory samples. No mixture
    # sampling or other episode is used in this pilot.
    heldout_frame = min(23, max_frame - 1)
    raw_examples = {frame: dataset[frame] for frame in set(selected + [heldout_frame])}
    heldout = copy.deepcopy(raw_examples[heldout_frame])
    asy_examples = []
    if run_async:
        training_alignment = alignment["training_mode"]
        training_latency = alignment["fixed_latency_steps"]
        asy_examples = _build_examples(
            dataset,
            selected,
            vlm_refresh_interval,
            asynchronous=True,
            alignment_mode=training_alignment,
            fixed_latency_steps=training_latency,
            trace_activation_steps=alignment["trace_activation_steps"],
        )
        heldout = _build_examples(
            dataset,
            [heldout_frame],
            vlm_refresh_interval,
            asynchronous=True,
            alignment_mode=training_alignment,
            fixed_latency_steps=training_latency,
            trace_activation_steps=alignment["trace_activation_steps"],
        )[0]

    base_cfg.framework.qwenvl.base_vlm = "/data/tzq/datasets/starVLA/playground/Pretrained_models/MiniCPM-V-4.6"
    base_cfg.framework.qwenvl.attn_implementation = "sdpa"
    base_cfg.framework.action_model.repeated_diffusion_steps = 1
    # This setting belongs only to the selected stage-2 async model. Stage 1
    # receives no refresh-cadence config and remains a synchronous baseline.
    if run_async:
        base_cfg.framework.vlm_refresh_interval = vlm_refresh_interval
        base_cfg.framework.async_alignment = {
            "mode": alignment["runtime_mode"],
            "fixed_latency_steps": alignment["fixed_latency_steps"],
            "trace_max_control_step": alignment["runtime_trace_max_control_step"],
            "trace_path": (
                str(pathlib.Path(alignment["runtime_trace_path"]).resolve())
                if alignment["runtime_trace_path"]
                else None
            ),
        }
    base_cfg.datasets.vla_data.obs_image_size = [224, 224]
    base_cfg.trainer.freeze_modules = "qwen_vl_interface,dino_encoder"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[pilot] dataset episode=0 task='put the bowl on the plate' frames={frame_count}")
    cadence_note = (
        f" vlm_refresh_interval={vlm_refresh_interval}"
        f" runtime_alignment={alignment['runtime_mode']}"
        f" training_alignment={alignment['training_mode']}"
        f" fixed_latency_steps={alignment['fixed_latency_steps']}"
        if run_async
        else " stage=1_synchronous"
    )
    print(
        f"[pilot] train_frames={selected} action_chunk_length={action_chunk_length}"
        f"{cadence_note} device={device}"
    )
    print(f"[pilot] results={output_dir}")

    dataset_path = args.data_root / DATASET_ID
    dataset_metadata = _dataset_metadata_manifest(dataset_path)
    (output_dir / "dataset_metadata_manifest.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n"
    )
    split_manifest = {
        "schema_version": 1,
        "scope": "single-trajectory engineering pilot; not a formal performance split",
        "task": "put the bowl on the plate",
        "train": [
            {"episode_index": 0, "frame_index": int(frame)} for frame in selected
        ],
        "validation": [{"episode_index": 0, "frame_index": int(heldout_frame)}],
        "test": [],
    }
    split_manifest["sha256"] = _canonical_json_hash(split_manifest)
    (output_dir / "data_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n"
    )

    config_hashes = {
        "training_config_path": str(args.config.resolve()),
        "training_config_sha256": _sha256_file(args.config),
        "async_config_path": str(args.async_config.resolve()) if run_async else None,
        "async_config_sha256": _sha256_file(args.async_config) if run_async else None,
        "training_alignment_trace_path": (
            alignment["training_trace_path"] if alignment else None
        ),
        "training_alignment_trace_sha256": (
            _sha256_file(alignment["training_trace_path"])
            if alignment and alignment["training_trace_path"]
            else None
        ),
    }
    run_provenance = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "run_arguments": {
            key: str(value) if isinstance(value, pathlib.Path) else value
            for key, value in vars(args).items()
        },
        "git": _git_provenance(),
        "configs": config_hashes,
        "dataset_metadata_manifest": dataset_metadata,
        "data_split_manifest_path": str(output_dir / "data_split_manifest.json"),
        "data_split_manifest_sha256": split_manifest["sha256"],
        "seed": int(args.seed),
        "checkpoint_initialization": {
            "vlm": str(base_cfg.framework.qwenvl.base_vlm),
            "action_head": "initialized by framework constructor before pilot updates",
            "saved_checkpoint_scope": "trainable parameters only",
        },
        "horizons": {
            "action_prediction_H": action_chunk_length,
            "execution_K": 1,
            "vlm_refresh_M": vlm_refresh_interval,
            "alignment_mode_runtime": alignment["runtime_mode"] if alignment else None,
            "alignment_mode_training": alignment["training_mode"] if alignment else None,
            "fixed_latency_L_steps": alignment["fixed_latency_steps"] if alignment else None,
        },
        "precision_policy": {
            "vlm_forward_autocast": "bfloat16",
            "action_model_forward_autocast": "float32",
            "reduce_in_full_precision": bool(
                base_cfg.framework.get("reduce_in_full_precision", False)
            ),
        },
        "gpu": _gpu_provenance(),
        "dependencies": _dependency_versions(),
        "simulator": {
            "python_path": str(LIBERO_PYTHON),
            "repository_path": str(LIBERO_ROOT),
            "skip_simulation": bool(args.skip_simulation),
        },
        "model_artifacts": {},
    }
    (output_dir / "run_provenance.json").write_text(
        json.dumps(run_provenance, indent=2) + "\n"
    )

    comparison = {
        "dataset": str(dataset_path),
        "provenance": run_provenance,
        "episode_index": 0,
        "task_instruction": "put the bowl on the plate",
        "episode_frames": frame_count,
        "training_frames": selected,
        "heldout_frame": heldout_frame,
        "action_horizon": action_chunk_length,
        "execution_horizon": 1,
        "vlm_refresh_interval": vlm_refresh_interval,
        "vlm_update_interval": vlm_refresh_interval,
        "runtime_alignment_mode": alignment["runtime_mode"] if alignment else None,
        "training_alignment_mode": alignment["training_mode"] if alignment else None,
        "fixed_latency_steps": alignment["fixed_latency_steps"] if alignment else None,
        "training_alignment_trace": alignment["training_trace_path"] if alignment else None,
        "training_vlm_alignment": (
            [
                {"control_step": int(frame), "vlm_anchor_step": int(example["vlm_anchor_frame"])}
                for frame, example in zip(selected, asy_examples)
            ]
            if run_async
            else None
        ),
        "heldout_vlm_anchor_step": heldout.get("vlm_anchor_frame"),
        "max_control_steps": args.max_control_steps,
        "models": {},
    }
    for framework_name in args.models:
        print(f"\n[pilot] ===== {framework_name} =====")
        cfg = copy.deepcopy(base_cfg)
        cfg.framework.name = framework_name
        resolved_config_text = OmegaConf.to_yaml(cfg, resolve=True)
        resolved_config_path = output_dir / f"{framework_name}_resolved_config.yaml"
        resolved_config_path.write_text(resolved_config_text, encoding="utf-8")
        resolved_config_sha256 = _sha256_bytes(resolved_config_text.encode("utf-8"))
        if framework_name == "MiniCPMGR00TDualAsy":
            examples = asy_examples
            eval_example = copy.deepcopy(heldout)
        else:
            examples = [copy.deepcopy(raw_examples[frame]) for frame in selected]
            eval_example = copy.deepcopy(heldout)
            eval_example.pop("vlm_image", None)
            eval_example.pop("vlm_anchor_frame", None)

        model = _new_model(framework_name, cfg, args.seed, device)
        training = _run_training(model, framework_name, examples, eval_example, args.learning_rate)
        checkpoint = _save_trainable_checkpoint(
            model,
            framework_name,
            output_dir,
            args.seed,
            resolved_config_sha256,
        )
        model_artifacts = {
            "resolved_config_path": str(resolved_config_path),
            "resolved_config_sha256": resolved_config_sha256,
            "checkpoint": checkpoint,
            "parameter_dtypes": {
                "vlm": _module_parameter_dtype(getattr(model, "qwen_vl_interface", None)),
                "dino": _module_parameter_dtype(getattr(model, "dino_encoder", None)),
                "action_model": _module_parameter_dtype(getattr(model, "action_model", None)),
            },
        }
        run_provenance["model_artifacts"][framework_name] = model_artifacts
        (output_dir / "run_provenance.json").write_text(
            json.dumps(run_provenance, indent=2) + "\n"
        )
        record = {"training": training, **model_artifacts}
        print(
            f"[pilot] train loss {training['train_losses'][0]['action_loss']:.6f} -> "
            f"{training['train_losses'][-1]['action_loss']:.6f}; "
            f"heldout={training['heldout_action_loss']:.6f}; "
            f"trainable={training['trainable_parameters'] / 1e6:.1f}M"
        )
        if not args.skip_simulation:
            simulation = _serve_and_simulate(
                model,
                framework_name,
                output_dir,
                "put the bowl on the plate",
                args.max_control_steps,
                args.seed,
            )
            record["simulation"] = simulation
            print(
                f"[pilot] simulation success={simulation['success']} "
                f"steps={simulation['control_steps']} "
                f"mean_policy={simulation['mean_policy_roundtrip_seconds']:.3f}s"
            )

        comparison["models"][framework_name] = record
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\n[pilot] completed: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
