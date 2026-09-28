# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Core implementation for the MiniCPM terminal DINO-residual framework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A.minicpm_video_history import (
    HistoryFrame,
    MiniCPMVideoHistoryAdapter,
    gather_token_hidden_states,
    valid_token_ids,
)
from starVLA.model.modules.action_model.DINOResidualHead import DINOResidualHead
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.modules.vlm import get_vlm_model


_LOCAL_MINICPM = Path("/data/tzq/datasets/starVLA/playground/Pretrained_models/MiniCPM-V-4.6")


def _as_config(config: Any) -> DictConfig:
    if config is None:
        raw: dict[str, Any] = {}
    elif isinstance(config, DictConfig):
        raw = OmegaConf.to_container(config, resolve=False) or {}
    elif isinstance(config, dict):
        raw = config
    else:
        raw = OmegaConf.to_container(OmegaConf.create(config), resolve=False) or {}

    cfg = OmegaConf.create(raw)
    if "framework" not in cfg or cfg.framework is None:
        cfg.framework = {}
    cfg.framework.name = "MiniCPMGR00TRes"
    defaults = OmegaConf.create(
        {
            "qwenvl": {
                "base_vlm": str(_LOCAL_MINICPM) if _LOCAL_MINICPM.is_dir() else "openbmb/MiniCPM-V-4.6",
                "attn_implementation": "sdpa",
                "enable_gradient_checkpointing": True,
            },
            "dino": {
                "dino_backbone": "dinov2_vits14",
                "image_size": 224,
                "camera_key": "video.primary_image",
            },
            "residual_model": {
                "num_residual_tokens": 256,
                "hidden_dim": 512,
                "num_residual_blocks": 2,
                "prediction_target": "residual",
                "goal_target": "successful_terminal",
                "text_loss_weight": 0.1,
                "max_context_tokens": 16384,
                "history_mode": "full",
                "video_fps": 10.0,
                "video_downsample_mode": "16x",
            },
            "runtime": {
                # This is the upper VLM refresh interval. It is independent of
                # any lower GR00T action_model.action_horizon.
                "execution_horizon": 8,
                "cache_mode": "recompute",
            },
        }
    )
    cfg.framework = OmegaConf.merge(defaults, cfg.framework)
    if "datasets" not in cfg or cfg.datasets is None:
        cfg.datasets = {}
    if "vla_data" not in cfg.datasets or cfg.datasets.vla_data is None:
        cfg.datasets.vla_data = {"obs_image_size": [224, 224]}
    if "trainer" not in cfg or cfg.trainer is None:
        cfg.trainer = {}
    return cfg


class MiniCPMGR00TResCore(baseframework):
    """Task-conditioned prediction of a successful terminal DINO feature delta.

    The target is defined by an explicit successful-terminal annotation, not a
    numeric frame offset. ``runtime.execution_horizon`` controls how often the
    upper VLM is refreshed. A lower action chunk length is deliberately not
    consumed by this model.
    """

    def __init__(self, config: Any = None, **kwargs) -> None:
        super().__init__()
        self.config = _as_config(config)
        target = str(self.config.framework.residual_model.prediction_target)
        if target not in {"residual", "absolute_goal"}:
            raise ValueError("residual_model.prediction_target must be 'residual' or 'absolute_goal'")
        if str(self.config.framework.residual_model.goal_target) != "successful_terminal":
            raise ValueError("stage 1 supports only explicit successful_terminal targets")
        cache_mode = str(self.config.framework.runtime.cache_mode)
        if cache_mode != "recompute":
            raise ValueError("only cache_mode='recompute' is enabled until prefix equivalence is validated")
        if int(self.config.framework.runtime.execution_horizon) < 1:
            raise ValueError("runtime.execution_horizon must be >= 1")

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.processor = self.qwen_vl_interface.processor
        self.tokenizer = self.processor.tokenizer
        self.residual_token = "<|dino_residual|>"
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.residual_token]})
        self.qwen_vl_interface.model.resize_token_embeddings(len(self.tokenizer))
        self.residual_token_id = int(self.tokenizer.convert_tokens_to_ids(self.residual_token))
        self.num_residual_tokens = int(self.config.framework.residual_model.num_residual_tokens)
        self.hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        if self.hidden_size != 1024:
            raise ValueError(f"MiniCPM-V hidden width must be 1024, got {self.hidden_size}")
        self.history_adapter = MiniCPMVideoHistoryAdapter(
            self.processor,
            residual_token=self.residual_token,
            num_residual_tokens=self.num_residual_tokens,
            fps=float(self.config.framework.residual_model.video_fps),
            video_downsample_mode=str(self.config.framework.residual_model.video_downsample_mode),
        )

        self.residual_head = DINOResidualHead(
            input_dim=self.hidden_size,
            hidden_dim=int(self.config.framework.residual_model.hidden_dim),
            output_dim=384,
            num_residual_blocks=int(self.config.framework.residual_model.num_residual_blocks),
            num_tokens=self.num_residual_tokens,
        )
        self.dino_encoder = get_dino_model(
            backone_name=str(self.config.framework.dino.dino_backbone)
        )
        if self.dino_encoder.num_channels != 384:
            raise ValueError("DINO residual target requires dinov2_vits14 patch width 384")
        if int(self.config.framework.dino.image_size) != 224:
            raise ValueError("the current residual target preprocessing is fixed at 224x224")
        for parameter in self.dino_encoder.parameters():
            parameter.requires_grad_(False)
        self.dino_encoder.eval()
        self.dino_teacher_sha256 = self._module_sha256(self.dino_encoder)
        self._freeze_micromamba_vision()

        self.execution_horizon = int(self.config.framework.runtime.execution_horizon)
        self.text_loss_weight = float(self.config.framework.residual_model.text_loss_weight)
        self.max_context_tokens = int(self.config.framework.residual_model.max_context_tokens)
        self.prediction_target = target
        self._episode_id: str | None = None
        self._instruction: str | None = None
        self._history: list[HistoryFrame] = []
        self._control_step = -1
        self._cached_goal: dict[str, Any] | None = None
        self._cached_refresh_step: int | None = None

    def train(self, mode: bool = True):
        """Train the LM/head while keeping frozen visual teachers in eval mode."""
        super().train(mode)
        self.dino_encoder.eval()
        self._freeze_micromamba_vision()
        return self

    def _freeze_micromamba_vision(self) -> None:
        """Freeze the MiniCPM visual tower and merger, leaving its LM trainable."""
        candidates = [self.qwen_vl_interface.model]
        candidates.append(getattr(self.qwen_vl_interface.model, "model", None))
        for owner in candidates:
            if owner is None:
                continue
            for attr in ("vision_tower", "visual", "merger", "vision_model"):
                module = getattr(owner, attr, None)
                if isinstance(module, nn.Module):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

    @property
    def vlm_base(self) -> nn.Module:
        """Underlying multimodal base model (without the LM logits projection)."""
        base = getattr(self.qwen_vl_interface.model, "model", None)
        if base is None:
            raise RuntimeError("MiniCPM conditional-generation model has no .model base")
        return base

    def _device(self) -> torch.device:
        return next(self.qwen_vl_interface.parameters()).device

    @staticmethod
    def _module_sha256(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _encode_vlm(self, batch: dict[str, Any], use_cache: bool = False):
        batch = batch.to(self._device()) if hasattr(batch, "to") else batch
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device().type == "cuda"):
            outputs = self.vlm_base(
                **batch,
                output_hidden_states=False,
                use_cache=use_cache,
                return_dict=True,
            )
        return outputs

    def _check_context_budget(self, inputs: dict[str, Any]) -> int:
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask")
        token_count = int(mask.sum().item()) if mask is not None else int(ids.shape[-1])
        if token_count > self.max_context_tokens:
            raise ValueError(
                f"MiniCPM history needs {token_count} tokens; configured budget is "
                f"{self.max_context_tokens}. No frames were silently truncated."
            )
        return token_count

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        transform = self.dino_encoder.dino_transform
        return transform(image.convert("RGB")).unsqueeze(0).to(self._device())

    @torch.no_grad()
    def encode_dino(self, image: Image.Image) -> torch.Tensor:
        self.dino_encoder.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device().type == "cuda"):
            return self.dino_encoder(self._image_tensor(image)).float()

    def _history_from_example(self, example: dict[str, Any]) -> tuple[list[HistoryFrame], str, str]:
        instruction = str(example["lang"])
        episode_id = str(example.get("episode_id", "train_episode"))
        frames = example.get("history", example.get("history_frames"))
        if frames is None:
            image = example.get("current_image")
            if image is None:
                images = example.get("image")
                if not images:
                    raise ValueError("example requires history/history_frames or current_image/image")
                image = images[0]
            frames = [
                HistoryFrame(
                    image=image.convert("RGB"),
                    episode_id=episode_id,
                    control_step=int(example.get("control_step", 0)),
                    timestamp_seconds=float(example.get("timestamp_seconds", 0.0)),
                    view_id=str(example.get("view_id", "primary")),
                    frame_order=int(example.get("frame_order", 0)),
                )
            ]
        return list(frames), instruction, episode_id

    def _forward_residual(self, example: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history, instruction, episode_id = self._history_from_example(example)
        current_image = example.get("current_image", history[-1].image)
        terminal_image = example.get("terminal_image", example.get("goal_image"))
        if example.get("goal_target_valid", True) is not True:
            raise ValueError("goal_target_valid must be true for residual supervision")
        if terminal_image is None:
            raise ValueError("residual supervision requires an explicit successful terminal_image")
        inputs = self.history_adapter.tokenize_history_query(
            instruction,
            episode_id,
            history,
            history_mode=str(self.config.framework.residual_model.history_mode),
            device=self._device(),
        )
        self._check_context_budget(inputs)
        outputs = self._encode_vlm(inputs, use_cache=False)
        query_hidden = gather_token_hidden_states(
            outputs.last_hidden_state,
            inputs["input_ids"],
            token_id=self.residual_token_id,
            expected_count=self.num_residual_tokens,
            attention_mask=inputs.get("attention_mask"),
        )
        with torch.autocast("cuda", dtype=torch.float32, enabled=self._device().type == "cuda"):
            predicted = self.residual_head(query_hidden.float())

        current = self.encode_dino(current_image)
        target_goal = self.encode_dino(terminal_image)
        target_residual = target_goal - current
        target = target_residual if self.prediction_target == "residual" else target_goal
        residual_loss = F.mse_loss(predicted.float(), target.float())
        predicted_residual = predicted.float() if self.prediction_target == "residual" else predicted.float() - current
        predicted_goal = current + predicted_residual
        return residual_loss, {
            "predicted_residual": predicted_residual,
            "target_residual": target_residual,
            "reference_features": current,
            "target_goal_features": target_goal,
            "predicted_goal_features": predicted_goal,
            "residual_token_hidden_states": query_hidden,
        }

    def _forward_text(self, example: dict[str, Any]) -> torch.Tensor:
        history, instruction, episode_id = self._history_from_example(example)
        assistant_text = str(example["assistant_text"])
        prefix, full = self.history_adapter.tokenize_history_with_text(
            instruction,
            episode_id,
            history,
            assistant_text,
            history_mode=str(self.config.framework.residual_model.history_mode),
            device=self._device(),
        )
        token_count = self._check_context_budget(full)
        prefix_ids = valid_token_ids(prefix["input_ids"], prefix.get("attention_mask"))[0]
        full_ids = valid_token_ids(full["input_ids"], full.get("attention_mask"))[0]
        if full_ids.numel() <= prefix_ids.numel() or not torch.equal(
            full_ids[: prefix_ids.numel()], prefix_ids
        ):
            raise ValueError("MiniCPM assistant prefix changed after appending the text label")
        outputs = self._encode_vlm(full, use_cache=False)
        answer_start = prefix_ids.numel()
        answer_logits = self.qwen_vl_interface.model.lm_head(
            outputs.last_hidden_state[:, answer_start - 1 : full_ids.numel() - 1]
        )
        labels = full_ids[answer_start:].to(answer_logits.device)
        if answer_logits.shape[1] != labels.numel() or labels.numel() == 0:
            raise RuntimeError(
                f"assistant text alignment failed: logits={answer_logits.shape[1]}, labels={labels.numel()}, "
                f"sequence_tokens={token_count}"
            )
        return F.cross_entropy(
            answer_logits.float().reshape(-1, answer_logits.shape[-1]),
            labels.reshape(-1),
        )

    def forward(self, examples: Sequence[dict[str, Any]], **kwargs) -> dict[str, torch.Tensor]:
        if isinstance(examples, dict):
            examples = [examples]
        if not examples:
            raise ValueError("MiniCPMGR00TRes.forward requires at least one example")
        residual_losses = []
        text_losses = []
        outputs = []
        for example in examples:
            residual_loss, output = self._forward_residual(example)
            residual_losses.append(residual_loss)
            outputs.append(output)
            if example.get("assistant_text") is not None:
                text_losses.append(self._forward_text(example))

        residual_loss = torch.stack(residual_losses).mean()
        text_loss = torch.stack(text_losses).mean() if text_losses else residual_loss.new_zeros(())
        total_loss = residual_loss + self.text_loss_weight * text_loss
        result = {
            "loss": total_loss,
            "total_loss": total_loss,
            "residual_loss": residual_loss,
            "text_loss": text_loss,
            "text_supervision_enabled": torch.tensor(bool(text_losses), device=total_loss.device),
        }
        # Preserve structured predictions for diagnostics without retaining all
        # per-example tensors in the optimizer's loss graph.
        result.update({k: torch.cat([out[k] for out in outputs], dim=0) for k in outputs[0]})
        return result

    def reset_episode(self, episode_id: str | None = None, instruction: str | None = None) -> None:
        """Clear all observation and plan state at an episode boundary."""
        self._episode_id = None if episode_id is None else str(episode_id)
        self._instruction = instruction
        self._history = []
        self._control_step = -1
        self._cached_goal = None
        self._cached_refresh_step = None

    def observe(
        self,
        image: Image.Image,
        control_step: int,
        timestamp_seconds: float,
        instruction: str,
        episode_id: str,
        view_id: str = "primary",
    ) -> dict[str, Any]:
        """Record every observed frame; refresh the goal at the configured cadence."""
        episode_id = str(episode_id)
        instruction = str(instruction)
        step = int(control_step)
        if self._episode_id is None:
            self.reset_episode(episode_id=episode_id, instruction=instruction)
        if episode_id != self._episode_id:
            raise ValueError("call reset_episode before observing a different episode")
        if self._history and step <= self._control_step:
            raise ValueError(f"control_step must increase, got {step} after {self._control_step}")
        instruction_changed = instruction != self._instruction
        self._instruction = instruction
        frame = HistoryFrame(
            image=image.convert("RGB"),
            episode_id=episode_id,
            control_step=step,
            timestamp_seconds=float(timestamp_seconds),
            view_id=str(view_id),
            frame_order=len(self._history),
        )
        self._history.append(frame)
        self._control_step = step
        should_refresh = (
            self._cached_goal is None
            or instruction_changed
            or step - int(self._cached_refresh_step) >= self.execution_horizon
        )
        if should_refresh:
            goal = self.predict_goal()
            self._cached_goal = goal
            self._cached_refresh_step = step
        else:
            goal = dict(self._cached_goal)
        goal["refreshed"] = bool(should_refresh)
        goal["plan_age"] = step - int(self._cached_refresh_step)
        goal["observed_control_step"] = step
        return goal

    @torch.inference_mode()
    def predict_goal(self) -> dict[str, Any]:
        """Predict the successful-terminal DINO residual from the full history."""
        if not self._history or self._episode_id is None or self._instruction is None:
            raise RuntimeError("reset_episode and observe at least one frame before predict_goal")
        inputs = self.history_adapter.tokenize_history_query(
            self._instruction,
            self._episode_id,
            self._history,
            history_mode=str(self.config.framework.residual_model.history_mode),
            device=self._device(),
        )
        self._check_context_budget(inputs)
        outputs = self._encode_vlm(inputs, use_cache=False)
        hidden = gather_token_hidden_states(
            outputs.last_hidden_state,
            inputs["input_ids"],
            token_id=self.residual_token_id,
            expected_count=self.num_residual_tokens,
            attention_mask=inputs.get("attention_mask"),
        )
        with torch.autocast("cuda", dtype=torch.float32, enabled=self._device().type == "cuda"):
            predicted = self.residual_head(hidden.float())
        reference = self.encode_dino(self._history[-1].image)
        residual = predicted.float() if self.prediction_target == "residual" else predicted.float() - reference
        goal = reference + residual
        return {
            "predicted_residual": residual,
            "reference_features": reference,
            "predicted_goal_features": goal,
            "residual_token_hidden_states": hidden,
            "reference_timestamp": float(self._history[-1].timestamp_seconds),
            "plan_age": 0,
            "history_frame_count": len(self._history),
            "control_step": self._control_step,
            "refreshed": True,
        }

    def save_reswam_checkpoint(self, directory: str | Path) -> None:
        """Save trainable parameters plus strict base/token/teacher metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        state = self.state_dict()
        trainable_state = {name: state[name].detach().cpu() for name in trainable_names}
        torch.save(trainable_state, directory / "trainable_state.pt")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        vocab_fingerprint = hashlib.sha256(
            json.dumps(self.tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        metadata = {
            "format_version": 1,
            "framework": "MiniCPMGR00TRes",
            "base_vlm": str(self.config.framework.qwenvl.base_vlm),
            "residual_token": self.residual_token,
            "residual_token_id": self.residual_token_id,
            "tokenizer_vocab_sha256": vocab_fingerprint,
            "num_residual_tokens": self.num_residual_tokens,
            "dino_backbone": str(self.config.framework.dino.dino_backbone),
            "dino_teacher_sha256": self.dino_teacher_sha256,
            "dino_preprocess": {
                "image_size": 224,
                "resize": "torchvision.Resize((224,224)), bilinear",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "prediction_target": self.prediction_target,
            "goal_target": str(self.config.framework.residual_model.goal_target),
            "execution_horizon": self.execution_horizon,
            "config": OmegaConf.to_container(self.config, resolve=False),
            "trainable_parameter_names": sorted(trainable_names),
        }
        (directory / "reswam_config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def load_reswam_checkpoint(self, directory: str | Path) -> None:
        """Restore only after verifying backbone, new token and teacher identity."""
        directory = Path(directory)
        metadata = json.loads((directory / "reswam_config.json").read_text())
        vocab_fingerprint = hashlib.sha256(
            json.dumps(self.tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        checks = {
            "framework": "MiniCPMGR00TRes",
            "base_vlm": str(self.config.framework.qwenvl.base_vlm),
            "residual_token": self.residual_token,
            "residual_token_id": self.residual_token_id,
            "tokenizer_vocab_sha256": vocab_fingerprint,
            "num_residual_tokens": self.num_residual_tokens,
            "dino_backbone": str(self.config.framework.dino.dino_backbone),
            "dino_teacher_sha256": self.dino_teacher_sha256,
            "prediction_target": self.prediction_target,
        }
        mismatches = {key: (metadata.get(key), value) for key, value in checks.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"checkpoint/model metadata mismatch: {mismatches}")
        saved_names = set(metadata.get("trainable_parameter_names", []))
        current_names = {name for name, p in self.named_parameters() if p.requires_grad}
        if saved_names != current_names:
            raise ValueError(
                "trainable parameter schema changed: "
                f"missing={sorted(current_names - saved_names)[:8]}, "
                f"unexpected={sorted(saved_names - current_names)[:8]}"
            )
        state = torch.load(directory / "trainable_state.pt", map_location="cpu", weights_only=True)
        if set(state) != saved_names:
            raise ValueError("trainable_state.pt keys do not match the checkpoint manifest")
        own = self.state_dict()
        for name, value in state.items():
            if own[name].shape != value.shape:
                raise ValueError(f"checkpoint shape mismatch for {name}: {tuple(value.shape)} vs {tuple(own[name].shape)}")
            own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))
