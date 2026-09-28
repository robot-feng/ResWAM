import importlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.MiniCPMGR00TDual import (
    MiniCPMGR00TDual,
    _copy_config_with_minicpm_defaults,
)
from starVLA.model.framework.VLM4A.MiniCPMGR00T import MiniCPM_GR00T
from starVLA.model.framework.VLM4A.QwenDual import Qwen_Dual
from starVLA.model.tools import FRAMEWORK_REGISTRY
from examples.modelExtensions.MiniCPM.libero_dual_pilot import _resolve_async_refresh_interval


def test_minicpm_gr00t_dual_registry_is_independent():
    assert FRAMEWORK_REGISTRY._registry["MiniCPMGR00TDual"] is MiniCPMGR00TDual
    assert FRAMEWORK_REGISTRY._registry["MiniCPMGR00T"] is MiniCPM_GR00T
    assert FRAMEWORK_REGISTRY._registry["QwenDual"] is Qwen_Dual


def test_minicpm_gr00t_dual_defaults_do_not_mutate_input():
    original = OmegaConf.create({"framework": {"name": "caller", "action_model": {"action_dim": 9}}})
    configured = _copy_config_with_minicpm_defaults(original)

    assert original.framework.name == "caller"
    assert configured.framework.name == "MiniCPMGR00TDual"
    assert "minicpm-v" in configured.framework.qwenvl.base_vlm.lower()
    assert configured.framework.qwenvl.attn_implementation == "sdpa"
    assert configured.framework.action_model.action_dim == 9
    assert configured.framework.action_model.num_target_vision_tokens == 32
    assert configured.datasets.vla_data.obs_image_size == [224, 224]


def test_stage1_pilot_does_not_load_async_config(tmp_path):
    missing_async_config = tmp_path / "no_async_config.yaml"
    assert _resolve_async_refresh_interval(
        ["MiniCPMGR00TDual"], None, missing_async_config
    ) is None
    with pytest.raises(ValueError, match="only valid when MiniCPMGR00TDualAsy is selected"):
        _resolve_async_refresh_interval(
            ["MiniCPMGR00TDual"], 8, missing_async_config
        )


def test_async_pilot_refresh_setting_is_explicit_and_validated(tmp_path):
    async_config = tmp_path / "async.yaml"
    async_config.write_text("framework:\n  vlm_refresh_interval: 6\n", encoding="utf-8")
    models = ["MiniCPMGR00TDual", "MiniCPMGR00TDualAsy"]
    assert _resolve_async_refresh_interval(models, None, async_config) == 6
    assert _resolve_async_refresh_interval(models, 3, async_config) == 3
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_async_refresh_interval(models, 0, async_config)


def test_minicpm_gr00t_dual_builds_joint_vlm_dino_condition(monkeypatch):
    qwen_dual_module = importlib.import_module("starVLA.model.framework.VLM4A.QwenDual")

    class DummyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=1024))

        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.zeros((len(images), 8), dtype=torch.long)}

        def forward(self, input_ids, **kwargs):
            hidden = self.anchor.expand(input_ids.shape[0], input_ids.shape[1], 1024)
            return SimpleNamespace(hidden_states=(hidden, hidden))

    class DummyDino(torch.nn.Module):
        num_channels = 384

        def prepare_dino_input(self, views):
            return torch.zeros((len(views), 3, 224, 224))

        def forward(self, image_tensor):
            return torch.ones((image_tensor.shape[0], 256, self.num_channels))

    class DummyAction(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            action_cfg = config.framework.action_model
            self.action_horizon = int(action_cfg.action_horizon)
            self.action_dim = int(action_cfg.action_dim)
            self.last_condition = None
            self.last_action_targets = None

        def forward(self, vl_embs, actions, state=None):
            self.last_condition = vl_embs
            self.last_action_targets = actions
            return self.anchor + vl_embs.float().square().mean() + actions.float().square().mean()

        def predict_action(self, vl_embs, state=None):
            return self.anchor.new_zeros(
                (vl_embs.shape[0], self.action_horizon, self.action_dim)
            )

    monkeypatch.setattr(qwen_dual_module, "get_vlm_model", lambda config: DummyVLM())
    monkeypatch.setattr(qwen_dual_module, "get_dino_model", lambda backone_name: DummyDino())
    monkeypatch.setattr(qwen_dual_module, "get_action_model", lambda config: DummyAction(config))

    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "MiniCPMGR00TDual",
                "action_model": {
                    "action_horizon": 3,
                    "action_dim": 4,
                    "state_dim": 4,
                    "repeated_diffusion_steps": 1,
                },
            }
        }
    )
    model = MiniCPMGR00TDual(cfg)
    image = Image.new("RGB", (224, 224), color="white")
    with torch.no_grad():
        model.qwen_vl_interface.anchor.fill_(3.0)
        model.dino_pro.weight.zero_()
        model.dino_pro.bias.fill_(2.0)
    example = {
        "image": [image],
        "lang": "move the object",
        "action": torch.ones((5, 4), dtype=torch.float32).numpy(),
    }
    condition, state = model.get_action_condition([[image]], [example["lang"]], None, None)
    training_output = model([example])
    training_output["action_loss"].backward()
    prediction_output = model.predict_action([example])

    assert condition.shape == (1, 8 + 256, 1024)
    assert torch.all(condition[:, :8] == 3.0)
    assert torch.all(condition[:, 8:] == 2.0)
    assert state is None
    assert training_output["action_loss"].ndim == 0
    assert model.action_model.last_action_targets.shape == (1, 3, 4)
    assert prediction_output["normalized_actions"].shape == (1, 3, 4)
    assert model.qwen_vl_interface.anchor.grad is not None
    assert model.dino_pro.bias.grad is not None
    assert torch.all(model.dino_pro.bias.grad != 0)
    assert not hasattr(model, "vlm_refresh_interval")
    assert not hasattr(model, "residual_head")
    assert model.config.framework.action_model.diffusion_model_cfg.cross_attention_dim == 1024
