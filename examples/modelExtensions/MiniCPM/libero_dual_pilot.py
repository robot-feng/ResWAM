"""Tiny same-trajectory training + one-task LIBERO pipeline comparison.

Compares frozen-VLM ``MiniCPMGR00T``, ``MiniCPMGR00TDual`` and
``MiniCPMGR00TDualAsy``. It uses only episode 0 from ``libero_goal`` for four
optimizer steps, then evaluates one 56-control-step rollout on the matching
LIBERO task. The VLM and DINO encoder are frozen; the action head and, for the
dual models, the DINO projection are trained.

The LIBERO simulator runs in the separate Python 3.12 ``libero`` environment.
The script serves each live policy over localhost TCP, so no checkpoint copy or
model installation is needed in that environment.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import json
import os
import pathlib
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

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = pathlib.Path("/data/tzq/datasets/starVLA/Datasets/libero_10hz")
DEFAULT_CONFIG = ROOT / "examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_dual_libero.yaml"
DEFAULT_ASY_CONFIG = ROOT / "examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_dual_asy_libero.yaml"
LIBERO_ROOT = pathlib.Path("/home/taizun/lcy/FastWAM/third_party/LIBERO-plus")
LIBERO_PYTHON = pathlib.Path("/data/miniconda3/envs/libero/bin/python")
MODEL_IDS = ("MiniCPMGR00T", "MiniCPMGR00TDual", "MiniCPMGR00TDualAsy")
TRAIN_FRAME_IDS = (1, 2, 9, 10)


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


def _build_examples(dataset, frame_ids, execution_horizon, asynchronous):
    examples = []
    for frame_id in frame_ids:
        example = copy.deepcopy(dataset[frame_id])
        if asynchronous:
            anchor_id = frame_id - frame_id % execution_horizon
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
        default=MODEL_IDS,
        help="model variants to run (defaults to the three-way comparison)",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help="low-level control steps between VLM refreshes (defaults independently to 8)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

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
    prediction_horizon = int(base_cfg.framework.action_model.action_horizon)
    async_cfg = OmegaConf.load(args.async_config)
    configured_execution_horizon = async_cfg.framework.action_model.get("execution_horizon")
    execution_horizon = args.execution_horizon
    if execution_horizon is None:
        execution_horizon = configured_execution_horizon
    if execution_horizon is None:
        execution_horizon = 8
    execution_horizon = int(execution_horizon)
    if prediction_horizon < 1:
        raise ValueError("framework.action_model.action_horizon must be >= 1")
    if execution_horizon < 1:
        raise ValueError("framework.action_model.execution_horizon must be >= 1")

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
    max_frame = min(frame_count - prediction_horizon, 47)
    selected = [int(x) for x in TRAIN_FRAME_IDS if x < max_frame]
    if len(selected) < args.train_steps:
        raise ValueError(f"episode 0 has too few valid frames for {args.train_steps} updates")
    selected = selected[: args.train_steps]

    # Materialize only the selected single-trajectory samples and their anchor
    # frames. No mixture sampling or other episode is used in this pilot.
    raw_examples = {frame: dataset[frame] for frame in set(selected + [16, 23]) if frame < max_frame}
    asy_examples = []
    for frame in selected:
        example = copy.deepcopy(raw_examples[frame])
        anchor = frame - frame % execution_horizon
        if anchor not in raw_examples:
            raw_examples[anchor] = dataset[anchor]
        example["vlm_image"] = copy.deepcopy(raw_examples[anchor]["image"])
        example["vlm_anchor_frame"] = anchor
        asy_examples.append(example)
    heldout_frame = min(23, max_frame - 1)
    heldout = copy.deepcopy(raw_examples.get(heldout_frame) or dataset[heldout_frame])
    heldout["vlm_image"] = copy.deepcopy(
        dataset[heldout_frame - heldout_frame % execution_horizon]["image"]
    )

    base_cfg.framework.qwenvl.base_vlm = "/data/tzq/datasets/starVLA/playground/Pretrained_models/MiniCPM-V-4.6"
    base_cfg.framework.qwenvl.attn_implementation = "sdpa"
    base_cfg.framework.action_model.repeated_diffusion_steps = 1
    # Keep this explicitly independent from action_horizon. The synchronous
    # baselines ignore it; the async model uses it for the VLM refresh cadence.
    base_cfg.framework.action_model.execution_horizon = execution_horizon
    base_cfg.datasets.vla_data.obs_image_size = [224, 224]
    base_cfg.trainer.freeze_modules = "qwen_vl_interface,dino_encoder"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[pilot] dataset episode=0 task='put the bowl on the plate' frames={frame_count}")
    print(
        f"[pilot] train_frames={selected} prediction_horizon={prediction_horizon} "
        f"execution_horizon={execution_horizon} VLM_refresh=every_{execution_horizon}_control_steps "
        f"device={device}"
    )
    print(f"[pilot] results={output_dir}")

    comparison = {
        "dataset": str(args.data_root / "libero_goal_no_noops_1.0.0_lerobot"),
        "episode_index": 0,
        "task_instruction": "put the bowl on the plate",
        "episode_frames": frame_count,
        "training_frames": selected,
        "heldout_frame": heldout_frame,
        "prediction_horizon": prediction_horizon,
        "execution_horizon": execution_horizon,
        "vlm_update_interval": execution_horizon,
        "max_control_steps": args.max_control_steps,
        "models": {},
    }
    for framework_name in args.models:
        print(f"\n[pilot] ===== {framework_name} =====")
        cfg = copy.deepcopy(base_cfg)
        cfg.framework.name = framework_name
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
        record = {"training": training}
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
