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
from examples.modelExtensions.MiniCPM.libero_dual_pilot import (
    _build_examples,
    _latest_completed_vlm_anchor_step,
    _resolve_async_alignment,
    _resolve_async_refresh_interval,
    _select_training_frames,
    _save_trainable_checkpoint,
)


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
    with pytest.raises(ValueError, match="only valid when MiniCPMGR00TDualAsy is selected"):
        _resolve_async_alignment(
            ["MiniCPMGR00TDual"], "synchronous", None, None, None, missing_async_config
        )


def test_async_pilot_refresh_setting_is_explicit_and_validated(tmp_path):
    async_config = tmp_path / "async.yaml"
    async_config.write_text(
        "framework:\n  vlm_refresh_interval: 6\n  async_alignment:\n    mode: fixed_step_delay\n    fixed_latency_steps: 3\n",
        encoding="utf-8",
    )
    models = ["MiniCPMGR00TDual", "MiniCPMGR00TDualAsy"]
    assert _resolve_async_refresh_interval(models, None, async_config) == 6
    assert _resolve_async_refresh_interval(models, 3, async_config) == 3
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_async_refresh_interval(models, 0, async_config)
    resolved = _resolve_async_alignment(models, None, None, None, None, async_config)
    assert resolved["runtime_mode"] == "fixed_step_delay"
    assert resolved["training_mode"] == "fixed_step_delay"
    assert resolved["fixed_latency_steps"] == 3
    overridden = _resolve_async_alignment(models, "fixed_step_delay", 2, None, None, async_config)
    assert overridden["fixed_latency_steps"] == 2
    no_mode_config = tmp_path / "no_mode.yaml"
    no_mode_config.write_text("framework:\n  vlm_refresh_interval: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires --fixed-latency-steps"):
        _resolve_async_alignment(models, "fixed_step_delay", None, None, None,
                                 no_mode_config)
    with pytest.raises(ValueError, match="no default VLM latency"):
        _resolve_async_alignment(models, None, None, None, None,
                                 no_mode_config)
    with pytest.raises(ValueError, match="only valid with fixed_step_delay"):
        _resolve_async_alignment(models, "wall_clock", 3, None, None,
                                 no_mode_config)


def test_wall_clock_training_alignment_requires_and_replays_an_observed_trace(tmp_path):
    async_config = tmp_path / "async.yaml"
    async_config.write_text("framework:\n  vlm_refresh_interval: 8\n", encoding="utf-8")
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"activation_events": [{"source_step": 0, "activation_step": 0}, '
        '{"source_step": 8, "activation_step": 11}]}',
        encoding="utf-8",
    )
    models = ["MiniCPMGR00TDualAsy"]
    with pytest.raises(ValueError, match="prior activation trace"):
        _resolve_async_alignment(models, "wall_clock", None, None, None, async_config)
    resolved = _resolve_async_alignment(
        models, "wall_clock", None, None, trace, async_config
    )
    assert resolved["runtime_mode"] == "wall_clock"
    assert resolved["training_mode"] == "trace_replay"
    assert resolved["trace_activation_steps"] == {0: 0, 8: 11}
    assert resolved["runtime_trace_max_control_step"] is None


def test_trace_replay_runtime_carries_trace_horizon(tmp_path):
    async_config = tmp_path / "async.yaml"
    async_config.write_text(
        "framework:\n  vlm_refresh_interval: 8\n", encoding="utf-8"
    )
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"async_step_trace": [{"control_step": 0, "cached_vlm_step": 0}, '
        '{"control_step": 11, "cached_vlm_step": 8}]}',
        encoding="utf-8",
    )
    resolved = _resolve_async_alignment(
        ["MiniCPMGR00TDualAsy"], "trace_replay", None, trace, None, async_config
    )
    assert resolved["runtime_trace_max_control_step"] == 11
    assert resolved["trace_activation_steps"] == {0: 0, 8: 11}


def test_async_training_anchor_tracks_latest_completed_refresh():
    anchor = lambda step: _latest_completed_vlm_anchor_step(step, 8, 3)
    assert [anchor(step) for step in (0, 7, 8, 9, 10, 11, 18, 19, 23)] == [
        0, 0, 0, 0, 0, 8, 8, 16, 16
    ]
    queued_anchor = lambda step: _latest_completed_vlm_anchor_step(step, 8, 9)
    assert [queued_anchor(step) for step in (8, 16, 17, 25, 26, 35)] == [
        0, 0, 8, 8, 16, 24
    ]

    dataset = [{"image": [step], "action": step} for step in range(24)]
    examples = _build_examples(
        dataset,
        [1, 2, 9, 10, 11, 19],
        8,
        True,
        alignment_mode="fixed_step_delay",
        fixed_latency_steps=3,
    )
    assert [example["vlm_anchor_frame"] for example in examples] == [0, 0, 0, 0, 8, 16]
    assert [example["vlm_image"] for example in examples] == [
        dataset[anchor]["image"] for anchor in (0, 0, 0, 0, 8, 16)
    ]
    with pytest.raises(ValueError, match="explicit alignment_mode"):
        _build_examples(dataset, [9], 8, True)


@pytest.mark.parametrize(
    ("latency", "expected_anchors"),
    [
        (0, [8, 8, 8, 8, 8]),
        (1, [0, 8, 8, 8, 8]),
        (2, [0, 0, 8, 8, 8]),
        (4, [0, 0, 0, 0, 8]),
    ],
)
def test_boundary_window_training_anchors_follow_controlled_latency(
    latency, expected_anchors
):
    dataset = [{"image": [step], "action": step} for step in range(24)]
    examples = _build_examples(
        dataset,
        [8, 9, 10, 11, 12],
        8,
        True,
        alignment_mode="fixed_step_delay",
        fixed_latency_steps=latency,
    )
    assert [example["vlm_anchor_frame"] for example in examples] == expected_anchors


def test_pilot_accepts_explicit_boundary_frames_for_alignment_ablation():
    boundary_window = [8, 9, 10, 11, 12]
    assert _select_training_frames(boundary_window, 5, max_frame=47) == boundary_window
    assert _select_training_frames(boundary_window, 3, max_frame=47) == [8, 9, 10]
    with pytest.raises(ValueError, match="must be in"):
        _select_training_frames([8, 47], 2, max_frame=47)
    with pytest.raises(ValueError, match="unique"):
        _select_training_frames([8, 8], 2, max_frame=47)
    with pytest.raises(ValueError, match="for 5 updates"):
        _select_training_frames([8, 9], 5, max_frame=47)


def test_synchronous_training_alignment_uses_current_periodic_refresh():
    dataset = [{"image": [step], "action": step} for step in range(24)]
    examples = _build_examples(
        dataset, [1, 8, 9, 16], 8, True, alignment_mode="synchronous"
    )
    assert [example["vlm_anchor_frame"] for example in examples] == [0, 8, 8, 16]


def test_pilot_saves_only_trainable_parameters_with_a_checksum(tmp_path):
    module = torch.nn.Linear(3, 2)
    module.bias.requires_grad_(False)

    checkpoint = _save_trainable_checkpoint(
        module, "test_policy", tmp_path, seed=7, config_sha256="cfg-hash"
    )

    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    assert set(payload["trainable_state_dict"]) == {"weight"}
    assert checkpoint["trainable_parameter_tensors"] == 1
    assert checkpoint["trainable_parameter_count"] == 6
    assert len(checkpoint["sha256"]) == 64


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
