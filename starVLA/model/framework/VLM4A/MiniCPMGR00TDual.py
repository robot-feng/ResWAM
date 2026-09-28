# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""MiniCPM-V + DINO dual-stream action model.

This framework reuses the independently implemented QwenDual architecture:
MiniCPM-V supplies language-conditioned sequence features, DINOv2 supplies
dense visual patch features, and the GR00T flow-matching head predicts actions.
The original ``MiniCPMGR00T`` and ``QwenDual`` registrations are untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from starVLA.model.framework.VLM4A.QwenDual import Qwen_Dual
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _copy_config_with_minicpm_defaults(config):
    """Copy an incoming config and select MiniCPM-V when no VLM is specified.

    Qwen_Dual owns the architecture defaults and merges ``config.framework``
    in-place. Clone first so constructing this model never mutates the caller's
    configuration object (which may also be shared with another model).
    """
    if config is None:
        raw = {}
    else:
        if hasattr(config, "unwrap"):
            config = config.unwrap()
        if isinstance(config, DictConfig):
            raw = OmegaConf.to_container(config, resolve=False)
        elif isinstance(config, dict):
            raw = copy.deepcopy(config)
        else:
            try:
                raw = OmegaConf.to_container(OmegaConf.create(config), resolve=False)
            except Exception as exc:
                raise TypeError(f"Unsupported MiniCPMGR00TDual config type: {type(config)!r}") from exc

    cfg = OmegaConf.create(raw or {})
    if "framework" not in cfg:
        cfg.framework = {}
    cfg.framework.name = "MiniCPMGR00TDual"
    if "qwenvl" not in cfg.framework or cfg.framework.qwenvl is None:
        cfg.framework.qwenvl = {}

    # Prefer the local MiniCPM-V 4.6 download used in this workspace, while
    # retaining the public HF id as a portable fallback.
    local_model = Path("/data/tzq/datasets/starVLA/playground/Pretrained_models/MiniCPM-V-4.6")
    default_model = str(local_model) if local_model.is_dir() else "openbmb/MiniCPM-V-4.6"
    if not cfg.framework.qwenvl.get("base_vlm"):
        cfg.framework.qwenvl.base_vlm = default_model
    if not cfg.framework.qwenvl.get("attn_implementation"):
        cfg.framework.qwenvl.attn_implementation = "sdpa"
    if "datasets" not in cfg or cfg.datasets is None:
        cfg.datasets = {}
    if "vla_data" not in cfg.datasets or cfg.datasets.vla_data is None:
        cfg.datasets.vla_data = {"obs_image_size": [224, 224]}
    elif "obs_image_size" not in cfg.datasets.vla_data:
        cfg.datasets.vla_data.obs_image_size = [224, 224]

    # QwenDual's dataclass predates several fields now required by the GR00T
    # flow-matching head. Fill only missing values here; explicit YAML values
    # remain authoritative and the shared QwenDual defaults stay untouched.
    if "action_model" not in cfg.framework or cfg.framework.action_model is None:
        cfg.framework.action_model = {}
    action_defaults = {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "action_horizon": 8,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
    }
    for key, value in action_defaults.items():
        if key not in cfg.framework.action_model or cfg.framework.action_model[key] is None:
            cfg.framework.action_model[key] = value
    return cfg


@FRAMEWORK_REGISTRY.register("MiniCPMGR00TDual")
class MiniCPMGR00TDual(Qwen_Dual):
    """MiniCPM-V 4.6 + DINOv2 + GR00T flow-matching action head.

    The framework name is intentionally distinct from ``MiniCPMGR00T``. The
    backbone is selected through the existing VLM dispatcher, so the shared
    MiniCPM-V processor and model implementation remain the single source of
    truth for MiniCPM inputs.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=_copy_config_with_minicpm_defaults(config), **kwargs)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        if hidden_size <= 0:
            raise ValueError(f"MiniCPM-V returned invalid hidden_size={hidden_size}")
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size


if __name__ == "__main__":
    import argparse

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser(description="MiniCPMGR00TDual local smoke test")
    parser.add_argument("--model_id", default=None, help="Optional HF id or local MiniCPM-V checkpoint path")
    parser.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "MiniCPMGR00TDual",
                "qwenvl": {
                    "base_vlm": args.model_id,
                    "attn_implementation": args.attn,
                },
                "dino": {"dino_backbone": "dinov2_vits14"},
                "action_model": {
                    "action_model_type": "DiT-B",
                    "action_hidden_dim": 1024,
                    "hidden_size": 1024,
                    "add_pos_embed": True,
                    "max_seq_len": 1024,
                    "action_dim": 7,
                    "state_dim": 7,
                    "action_horizon": 8,
                    "repeated_diffusion_steps": 2,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 32,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 1024,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
            },
            "trainer": {"repeated_diffusion_steps": 2},
            "datasets": {"vla_data": {"obs_image_size": None}},
        }
    )
    model = MiniCPMGR00TDual(cfg)
    print(f"MiniCPM hidden size: {model.qwen_vl_interface.model.config.hidden_size}")

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    example = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image],
        "lang": "Move the cup next to the plate.",
    }
    model.eval()
    with torch.no_grad():
        loss = model([example])["action_loss"]
        actions = model.predict_action([example])["normalized_actions"]
    print(f"action_loss={float(loss):.6f}; predicted_actions={actions.shape}")
