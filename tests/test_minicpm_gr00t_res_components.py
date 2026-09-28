from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

import starVLA.dataloader.minicpm_res_lerobot as residual_data
from starVLA.dataloader.minicpm_res_lerobot import load_assistant_labels, load_success_terminals
from starVLA.dataloader.minicpm_res_splits import (
    build_residual_split_manifest,
    load_residual_episode_split,
    validate_residual_split_manifest,
)
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


def test_residual_target_alignment_zero_case_and_future_target_isolation():
    captured = {}

    class _HistoryAdapter:
        def tokenize_history_query(self, instruction, episode_id, history, history_mode, device):
            captured["history"] = list(history)
            captured["instruction"] = instruction
            captured["episode_id"] = episode_id
            return {
                "input_ids": torch.full((1, 256), 77, dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, 256), dtype=torch.long, device=device),
            }

    class _ConstantResidualHead(torch.nn.Module):
        def forward(self, query_hidden_states):
            return torch.full(
                (query_hidden_states.shape[0], 256, 384),
                0.25,
                device=query_hidden_states.device,
            )

    model = object.__new__(MiniCPMGR00TResCore)
    torch.nn.Module.__init__(model)
    model.history_adapter = _HistoryAdapter()
    model.residual_head = _ConstantResidualHead()
    model.num_residual_tokens = 256
    model.residual_token_id = 77
    model.prediction_target = "residual"
    model.config = types.SimpleNamespace(
        framework=types.SimpleNamespace(
            residual_model=types.SimpleNamespace(history_mode="full")
        )
    )
    model._device = types.MethodType(lambda self: torch.device("cpu"), model)
    model._history_from_example = types.MethodType(
        lambda self, example: (example["history"], example["lang"], example["episode_id"]),
        model,
    )
    model._check_context_budget = types.MethodType(lambda self, inputs: 256, model)
    model._encode_vlm = types.MethodType(
        lambda self, inputs, use_cache: types.SimpleNamespace(
            last_hidden_state=torch.zeros((1, 256, 1024), device=inputs["input_ids"].device)
        ),
        model,
    )
    current = Image.new("RGB", (8, 8), color=(1, 2, 3))
    future = Image.new("RGB", (8, 8), color=(4, 5, 6))
    feature_by_image_id = {
        id(current): torch.ones((1, 256, 384)),
        id(future): torch.full((1, 256, 384), 3.0),
    }
    model.encode_dino = types.MethodType(
        lambda self, image: feature_by_image_id[id(image)], model
    )
    observed = HistoryFrame(current, "episode-a", 4, 0.4, "primary", 4)
    example = {
        "history": [observed],
        "current_image": current,
        "terminal_image": future,
        "lang": "move the object",
        "episode_id": "episode-a",
        "goal_target_valid": True,
    }

    loss, outputs = model._forward_residual(example)
    assert captured["history"] == [observed]
    assert all(frame.image is not future for frame in captured["history"])
    assert captured["episode_id"] == "episode-a"
    assert torch.all(outputs["target_residual"] == 2.0)
    assert torch.all(outputs["predicted_residual"] == 0.25)
    assert torch.all(outputs["predicted_goal_features"] == 1.25)
    assert loss.item() == pytest.approx((2.0 - 0.25) ** 2)

    # At the annotated terminal step current and goal features are identical,
    # so the target residual must be exactly zero.
    zero_example = {**example, "terminal_image": current}
    _, zero_outputs = model._forward_residual(zero_example)
    assert torch.count_nonzero(zero_outputs["target_residual"]) == 0


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


def test_vlm_refresh_interval_is_independent_of_action_horizon():
    cfg = _as_config(
        {
            "framework": {
                "action_model": {"action_horizon": 3},
                "runtime": {"vlm_refresh_interval": 8},
            }
        }
    )
    assert cfg.framework.action_model.action_horizon == 3
    assert cfg.framework.runtime.vlm_refresh_interval == 8
    assert cfg.framework.residual_model.goal_target == "successful_terminal"
    assert cfg.framework.residual_model.prediction_target == "residual"

    async_cfg = _asy_config(
        {"framework": {"action_model": {"action_horizon": 3}}}
    )
    assert async_cfg.framework.action_model.action_horizon == 3
    assert async_cfg.framework.vlm_refresh_interval == 8
    # The current MiniCPM DualAsy pilot does not implement a configurable
    # execution horizon. Its evaluator commits chunk[0] and requests a new
    # action on the next control step, so do not treat an ignored config key
    # as proof that action execution length is configurable.
    assert "execution_horizon" not in async_cfg.framework.action_model
    explicit_cfg = _asy_config(
        {
            "framework": {
                "vlm_refresh_interval": 2,
                "action_model": {"action_horizon": 3},
            }
        }
    )
    assert explicit_cfg.framework.vlm_refresh_interval == 2
    assert explicit_cfg.framework.action_model.action_horizon == 3

    with pytest.raises(ValueError, match="controls action execution, not VLM refresh"):
        _as_config({"framework": {"runtime": {"execution_horizon": 6}}})


def test_prefix_cache_is_opt_in_and_has_a_decoded_output_tolerance():
    cfg = _as_config({"framework": {"runtime": {"cache_mode": "prefix"}}})
    assert cfg.framework.runtime.cache_mode == "prefix"
    assert cfg.framework.runtime.prefix_cache_validate_every == 8
    assert cfg.framework.runtime.prefix_cache_max_relative_rms == pytest.approx(0.05)
    assert _as_config(None).framework.runtime.cache_mode == "recompute"


def test_streaming_refresh_cadence_and_keeps_all_frames():
    model = object.__new__(MiniCPMGR00TResCore)
    torch.nn.Module.__init__(model)
    model.vlm_refresh_interval = 8
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
    model.vlm_refresh_interval = 8
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
    model.vlm_refresh_interval = 8
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


def test_residual_dataset_episode_and_control_step_filters(tmp_path, monkeypatch):
    manifest = tmp_path / "terminals.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "0", "terminal_step": 15, "is_success": True}),
                json.dumps({"episode_id": "1", "terminal_step": 15, "is_success": True}),
                json.dumps({"episode_id": "2", "terminal_step": 15, "is_success": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_dataset = types.SimpleNamespace(
        all_steps=[(episode, step) for episode in range(3) for step in (0, 8)],
        modality_keys={"video": ["video.primary_image"]},
        trajectory_ids=[0, 1, 2],
        trajectory_lengths=[16, 16, 16],
    )
    monkeypatch.setattr(residual_data, "make_LeRobotSingleDataset", lambda *args, **kwargs: fake_dataset)
    selected = residual_data.MiniCPMResidualLeRobotDataset(
        data_root_dir=Path(tmp_path),
        dataset_name="fake",
        robot_type="fake",
        success_terminal_manifest=manifest,
        camera_key="video.primary_image",
        sample_stride=8,
        episode_ids=["0", "1"],
        control_steps=[8],
    )
    assert selected._steps == [(0, 8), (1, 8)]
    with pytest.raises(ValueError, match="lack explicit successful-terminal labels"):
        residual_data.MiniCPMResidualLeRobotDataset(
            data_root_dir=Path(tmp_path),
            dataset_name="fake",
            robot_type="fake",
            success_terminal_manifest=manifest,
            camera_key="video.primary_image",
            sample_stride=8,
            episode_ids=["2"],
        )


def test_residual_split_manifest_pins_legacy_validation_and_blocks_group_leakage(tmp_path):
    dataset_path = tmp_path / "fake_dataset"
    meta_path = dataset_path / "meta"
    meta_path.mkdir(parents=True)
    task_by_id = {
        "0": "task-a",
        "1": "task-a",
        "2": "task-a",
        "3": "task-a",
        "4": "task-b",
        "5": "task-b",
        "6": "task-b",
        "7": "task-b",
        "8": "task-a",
        "9": "task-b",
    }
    (meta_path / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": int(episode_id),
                    "tasks": [task],
                    "length": 4,
                }
            )
            + "\n"
            for episode_id, task in task_by_id.items()
        ),
        encoding="utf-8",
    )

    annotation_path = tmp_path / "terminals.jsonl"
    rows = []
    for episode_id, task in task_by_id.items():
        if episode_id == "9":
            rows.append(
                {
                    "episode_id": episode_id,
                    "task": task,
                    "is_success": False,
                    "reason": "ambiguous",
                }
            )
            continue
        source_demo = "demo_0" if episode_id in ("0", "8") else f"demo_{episode_id}"
        source_file = "task_a.hdf5" if task == "task-a" else "task_b.hdf5"
        rows.append(
            {
                "episode_id": episode_id,
                "task": task,
                "terminal_step": 3,
                "is_success": True,
                "provenance": {
                    "source_file": source_file,
                    "source_demo": source_demo,
                },
            }
        )
    annotation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    options = {
        "success_terminal_manifest": annotation_path,
        "dataset_path": dataset_path,
        "success_manifest_repo_path": "terminals.jsonl",
        "seed": 17,
        "legacy_train_episode_ids": ("0",),
        "legacy_validation_episode_ids": ("1",),
    }
    manifest = build_residual_split_manifest(**options)
    assert manifest == build_residual_split_manifest(**options)
    validate_residual_split_manifest(
        manifest,
        success_terminal_manifest=annotation_path,
        dataset_path=dataset_path,
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(manifest), encoding="utf-8")
    train = load_residual_episode_split(
        split_path,
        split="train",
        success_terminal_manifest=annotation_path,
        dataset_path=dataset_path,
    )
    validation = load_residual_episode_split(
        split_path,
        split="validation",
        success_terminal_manifest=annotation_path,
        dataset_path=dataset_path,
    )
    test = load_residual_episode_split(
        split_path,
        split="test",
        success_terminal_manifest=annotation_path,
        dataset_path=dataset_path,
    )
    assert "0" in train["episode_ids"] and "8" in train["episode_ids"]
    assert "1" in validation["episode_ids"]
    assert set(train["episode_ids"]).isdisjoint(validation["episode_ids"])
    assert set(train["episode_ids"]).isdisjoint(test["episode_ids"])
    assert set(validation["episode_ids"]).isdisjoint(test["episode_ids"])
    assert manifest["counts"]["excluded_unlabeled_episodes"] == 1
    assert manifest["counts"]["source_demo_groups"] == 8

    annotation_path.write_text(annotation_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation hash"):
        load_residual_episode_split(
            split_path,
            split="test",
            success_terminal_manifest=annotation_path,
            dataset_path=dataset_path,
        )


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
