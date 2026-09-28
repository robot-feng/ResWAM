from __future__ import annotations

import json
import types

import pytest
import torch
from PIL import Image

from starVLA.dataloader.minicpm_res_lerobot import load_assistant_labels, load_success_terminals
from starVLA.model.framework.VLM4A.minicpm_gr00t_res_core import _as_config
from starVLA.model.framework.VLM4A.minicpm_gr00t_res_core import MiniCPMGR00TResCore
from starVLA.model.framework.VLM4A.MiniCPMGR00TDualAsy import _asy_config
from starVLA.model.framework.VLM4A.minicpm_video_history import HistoryFrame, gather_token_hidden_states
from starVLA.model.modules.action_model.DINOResidualHead import DINOResidualHead


def test_residual_head_shapes_and_gradients():
    head = DINOResidualHead()
    query = torch.randn(2, 256, 1024, requires_grad=True)
    prediction = head(query)
    assert prediction.shape == (2, 256, 384)
    prediction.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_gather_query_tokens_with_left_padding():
    token_id = 77
    ids = torch.tensor([[0, 5, token_id, token_id, 0], [6, token_id, token_id, 7, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    result = gather_token_hidden_states(hidden, ids, token_id, expected_count=2, attention_mask=mask)
    assert result.shape == (2, 2, 3)
    assert torch.equal(result[0], hidden[0, 2:4])
    assert torch.equal(result[1], hidden[1, 1:3])


def test_history_frame_requires_explicit_valid_metadata():
    image = Image.new("RGB", (32, 32))
    frame = HistoryFrame(image, "episode-a", 2, 0.2, "primary", 2)
    assert frame.view_id == "primary"
    with pytest.raises(ValueError, match="nonnegative"):
        HistoryFrame(image, "episode-a", -1, 0.2, "primary", 0)
    with pytest.raises(ValueError, match="timestamp"):
        HistoryFrame(image, "episode-a", 0, float("nan"), "primary", 0)


def test_execution_horizon_is_independent_of_action_horizon():
    cfg = _as_config(
        {
            "framework": {
                "action_model": {"action_horizon": 3},
                "runtime": {"execution_horizon": 8},
            }
        }
    )
    assert cfg.framework.action_model.action_horizon == 3
    assert cfg.framework.runtime.execution_horizon == 8
    assert cfg.framework.residual_model.goal_target == "successful_terminal"
    assert cfg.framework.residual_model.prediction_target == "residual"

    async_cfg = _asy_config(
        {"framework": {"action_model": {"action_horizon": 3}}}
    )
    assert async_cfg.framework.action_model.action_horizon == 3
    assert async_cfg.framework.action_model.execution_horizon == 8
    async_cfg = _asy_config(
        {"framework": {"action_model": {"action_horizon": 3, "execution_horizon": 5}}}
    )
    assert async_cfg.framework.action_model.execution_horizon == 5


def test_prefix_cache_is_opt_in_and_has_a_decoded_output_tolerance():
    cfg = _as_config({"framework": {"runtime": {"cache_mode": "prefix"}}})
    assert cfg.framework.runtime.cache_mode == "prefix"
    assert cfg.framework.runtime.prefix_cache_validate_every == 8
    assert cfg.framework.runtime.prefix_cache_max_relative_rms == pytest.approx(0.05)
    assert _as_config(None).framework.runtime.cache_mode == "recompute"


def test_streaming_refresh_cadence_and_keeps_all_frames():
    model = object.__new__(MiniCPMGR00TResCore)
    torch.nn.Module.__init__(model)
    model.execution_horizon = 8
    model._episode_id = None
    model._instruction = None
    model._history = []
    model._control_step = -1
    model._cached_goal = None
    model._cached_refresh_step = None
    refreshes = []

    def fake_predict_goal(self):
        refreshes.append(self._control_step)
        return {"cached_step": self._control_step}

    model.predict_goal = types.MethodType(fake_predict_goal, model)
    image = Image.new("RGB", (8, 8))
    outputs = [
        model.observe(image, step, step / 10, "move the mug", "episode-a")
        for step in range(17)
    ]
    assert refreshes == [0, 8, 16]
    assert len(model._history) == 17
    assert outputs[7]["plan_age"] == 7
    assert outputs[7]["history_frame_count"] == 8
    assert outputs[7]["plan_history_frame_count"] == 1
    assert outputs[8]["plan_age"] == 0
    assert outputs[8]["history_frame_count"] == 9
    assert outputs[8]["plan_history_frame_count"] == 9
    assert outputs[8]["refreshed"] is True


def test_streaming_rejects_episode_switch_without_reset():
    model = object.__new__(MiniCPMGR00TResCore)
    torch.nn.Module.__init__(model)
    model.execution_horizon = 8
    model._episode_id = None
    model._instruction = None
    model._history = []
    model._control_step = -1
    model._cached_goal = None
    model._cached_refresh_step = None
    model.predict_goal = types.MethodType(lambda self: {"cached_step": self._control_step}, model)
    image = Image.new("RGB", (8, 8))
    model.observe(image, 0, 0.0, "task", "episode-a")
    with pytest.raises(ValueError, match="reset_episode"):
        model.observe(image, 1, 0.1, "task", "episode-b")


def test_reset_episode_discards_hybrid_cache_and_reenables_validation():
    model = object.__new__(MiniCPMGR00TResCore)
    torch.nn.Module.__init__(model)
    model.execution_horizon = 8
    model._episode_id = "episode-a"
    model._instruction = "task"
    model._history = [HistoryFrame(Image.new("RGB", (8, 8)), "episode-a", 0, 0, "primary", 0)]
    model._control_step = 0
    model._cached_goal = {"some": "goal"}
    model._cached_refresh_step = 0
    model._prefix_cache = object()
    model._prefix_cache_token_count = 100
    model._prefix_cache_history_count = 1
    model._prefix_cache_instruction = "task"
    model._prefix_cache_refresh_count = 4
    model._prefix_cache_validated = True
    model._prefix_cache_disabled_reason = "stale"
    model._prefix_cache_last_metrics = {"relative_rms": 0.01}

    model.reset_episode("episode-b", "new task")

    assert model._history == []
    assert model._prefix_cache is None
    assert model._prefix_cache_token_count == 0
    assert model._prefix_cache_history_count == 0
    assert model._prefix_cache_instruction is None
    assert model._prefix_cache_refresh_count == 0
    assert model._prefix_cache_validated is False
    assert model._prefix_cache_disabled_reason is None
    assert model._prefix_cache_last_metrics is None


def test_prefix_cache_validation_returns_recompute_output_and_falls_back_when_error_is_large():
    model = object.__new__(MiniCPMGR00TResCore)
    torch.nn.Module.__init__(model)
    model._prefix_cache_refresh_count = 0
    model.prefix_cache_validate_every = 8
    model.prefix_cache_max_relative_rms = 0.05
    model._prefix_cache_validated = False
    model._prefix_cache_last_metrics = None
    model._prefix_cache = object()
    model._prefix_cache_disabled_reason = None

    def cached_hidden(self):
        return torch.tensor([1.01, 0.99])

    def decode(self, hidden):
        return {"predicted_residual": hidden}

    def recompute(self):
        return {"predicted_residual": torch.ones(2)}

    model._cached_query_hidden = types.MethodType(cached_hidden, model)
    model._decode_goal = types.MethodType(decode, model)
    model._predict_goal_recompute = types.MethodType(recompute, model)
    result = model._predict_goal_with_prefix_cache()
    assert result["cache_mode"] == "prefix_validated"
    assert result["cache_relative_rms_error"] == pytest.approx(0.01, abs=1e-5)
    assert torch.equal(result["predicted_residual"], torch.ones(2))

    model._prefix_cache_refresh_count = 0
    model._prefix_cache_validated = False
    model._prefix_cache = object()
    model.prefix_cache_max_relative_rms = 0.05
    model._cached_query_hidden = types.MethodType(lambda self: torch.tensor([1.2, 0.8]), model)
    with pytest.warns(RuntimeWarning, match="falling back"):
        result = model._predict_goal_with_prefix_cache()
    assert result["cache_mode"] == "recompute_fallback"
    assert model._prefix_cache_disabled_reason is not None


def test_success_manifest_never_infers_final_episode_frame(tmp_path):
    manifest = tmp_path / "terminals.jsonl"
    manifest.write_text(
        json.dumps({"episode_id": "3", "terminal_step": 12, "is_success": True}) + "\n"
        + json.dumps({"episode_id": "4", "terminal_step": 9, "is_success": False}) + "\n",
        encoding="utf-8",
    )
    assert load_success_terminals(manifest) == {"3": 12}
    with pytest.raises(ValueError, match="no successful-terminal annotations"):
        empty = tmp_path / "empty.jsonl"
        empty.write_text(json.dumps({"episode_id": "4", "terminal_step": 9, "is_success": False}) + "\n")
        load_success_terminals(empty)


def test_assistant_manifest_keys_and_duplicate_validation(tmp_path):
    manifest = tmp_path / "assistant.jsonl"
    manifest.write_text(
        json.dumps({"episode_id": "3", "control_step": 5, "text": "The mug is on the plate."}) + "\n",
        encoding="utf-8",
    )
    assert load_assistant_labels(manifest) == {("3", 5): "The mug is on the plate."}
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "3", "control_step": 5, "text": "a"}),
                json.dumps({"episode_id": "3", "control_step": 5, "text": "b"}),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_assistant_labels(manifest)
