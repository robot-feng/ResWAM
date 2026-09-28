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
    assert outputs[8]["plan_age"] == 0
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
