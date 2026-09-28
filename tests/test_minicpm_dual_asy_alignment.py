import json
import queue

import pytest
import torch

from starVLA.model.framework.VLM4A.MiniCPMGR00TDualAsy import MiniCPMGR00TDualAsy
from starVLA.model.framework.VLM4A.minicpm_dual_asy_alignment import (
    fixed_activation_steps,
    load_trace_activation_steps,
    observed_activation_steps,
    source_step_at,
)


@pytest.mark.parametrize(
    ("latency", "expected"),
    [
        (0, {0: 0, 8: 8, 16: 16, 24: 24}),
        (1, {0: 0, 8: 9, 16: 17, 24: 25}),
        (2, {0: 0, 8: 10, 16: 18, 24: 26}),
        (4, {0: 0, 8: 12, 16: 20, 24: 28}),
    ],
)
def test_fixed_schedule_maps_requested_source_to_activation(latency, expected):
    assert fixed_activation_steps(8, latency, 24) == expected


def test_fixed_schedule_models_worker_queue_when_latency_exceeds_refresh_interval():
    assert fixed_activation_steps(4, 6, 12) == {0: 0, 4: 10, 8: 16, 12: 22}


@pytest.mark.parametrize(
    ("latency", "checks"),
    [
        (0, [(7, 0), (8, 8), (15, 8), (16, 16)]),
        (1, [(8, 0), (9, 8), (16, 8), (17, 16)]),
        (2, [(9, 0), (10, 8), (17, 8), (18, 16)]),
        (4, [(11, 0), (12, 8), (19, 8), (20, 16)]),
    ],
)
def test_training_anchor_uses_latest_source_available_at_control_step(latency, checks):
    for step, expected in checks:
        assert source_step_at(
            step,
            mode="fixed_step_delay",
            refresh_interval=8,
            fixed_latency_steps=latency,
        ) == expected


def test_synchronous_mode_is_fixed_refresh_with_zero_delivery_delay():
    assert [
        source_step_at(step, mode="synchronous", refresh_interval=4)
        for step in range(9)
    ] == [0, 0, 0, 0, 4, 4, 4, 4, 8]


def test_wall_clock_cannot_be_guessed_for_offline_training_anchors():
    with pytest.raises(ValueError, match="cannot predict wall-clock readiness"):
        source_step_at(9, mode="wall_clock", refresh_interval=8)


def test_trace_replay_reads_explicit_activation_events(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "activation_events": [
                    {"source_step": 0, "activation_step": 0},
                    {"source_step": 8, "activation_step": 11},
                    {"source_step": 16, "activation_step": 19},
                ]
            }
        ),
        encoding="utf-8",
    )
    events = load_trace_activation_steps(trace)
    assert events == {0: 0, 8: 11, 16: 19}
    assert source_step_at(
        18, mode="trace_replay", refresh_interval=8, trace_activation_steps=events
    ) == 8
    assert source_step_at(
        19, mode="trace_replay", refresh_interval=8, trace_activation_steps=events
    ) == 16


def test_trace_replay_extracts_first_use_of_each_source():
    trace = [
        {"control_step": 0, "cached_vlm_step": 0},
        {"control_step": 1, "cached_vlm_step": 0},
        {"control_step": 11, "cached_vlm_step": 8},
        {"control_step": 12, "cached_vlm_step": 8},
    ]
    assert observed_activation_steps(trace) == {0: 0, 8: 11}


def test_trace_replay_requires_step_zero_bootstrap():
    with pytest.raises(ValueError, match="bootstrap"):
        source_step_at(
            2,
            mode="trace_replay",
            refresh_interval=8,
            trace_activation_steps={8: 9},
        )


def _stub_async_policy(mode):
    policy = object.__new__(MiniCPMGR00TDualAsy)
    torch.nn.Module.__init__(policy)
    policy.async_alignment_mode = mode
    policy._async_generation = 1
    policy._control_step = 12
    policy._cached_vlm_hidden = None
    policy._cached_instruction = None
    policy._cached_vlm_step = None
    policy._cached_ready_timestamp = None
    policy._completed_snapshots = {}
    policy._activation_events = []
    policy._result_queue = queue.Queue()
    policy._request_queue = queue.Queue()
    policy._stats = {"vlm_completed": 0, "vlm_dropped_stale": 0, "vlm_seconds": []}
    return policy


def test_controlled_runtime_publishes_only_at_scheduled_activation_step():
    policy = _stub_async_policy("trace_replay")
    hidden = torch.ones((1, 2, 3))
    policy._result_queue.put(
        {
            "generation": 1,
            "source_step": 8,
            "instructions": ("task",),
            "hidden": hidden,
            "compute_seconds": 0.25,
            "request_timestamp": 10.0,
            "ready_timestamp": 10.25,
            "error": None,
        }
    )

    policy._wait_for_scheduled_source(8, activation_step=11)

    assert policy._cached_vlm_step == 8
    assert policy._activation_events[0]["activation_step"] == 11
    assert policy._activation_events[0]["request_timestamp"] == 10.0
    assert policy._activation_events[0]["ready_timestamp"] == 10.25


def test_wall_clock_activation_uses_first_action_boundary_after_ready():
    policy = _stub_async_policy("wall_clock")
    policy._result_queue.put(
        {
            "generation": 1,
            "source_step": 8,
            "instructions": ("task",),
            "hidden": torch.ones((1, 2, 3)),
            "compute_seconds": 0.25,
            "request_timestamp": 10.0,
            "ready_timestamp": 10.25,
            "error": None,
        }
    )

    policy._poll_vlm_results(activation_step=12)

    assert policy._cached_vlm_step == 8
    assert policy._activation_events[0]["activation_step"] == 12
    assert policy._activation_events[0]["activation_timestamp"] >= 10.25


def test_wall_clock_action_uses_old_cache_without_waiting_for_refresh():
    policy = _stub_async_policy("wall_clock")
    policy._control_step = 9
    policy._cached_instruction = ("task",)
    policy._cached_vlm_hidden = torch.ones((1, 2, 3))
    policy._cached_vlm_step = 0
    policy.vlm_update_interval = 8
    submitted = []
    policy.align_model_input = lambda examples: ([], None, ["task"], None)
    policy._submit_vlm_refresh = lambda *args: submitted.append(args)
    policy._poll_vlm_results = lambda activation_step=None: None
    policy._condition_from_vlm_hidden = lambda *args: (torch.ones((1, 2, 3)), None)

    class ActionHead(torch.nn.Module):
        def predict_action(self, condition, state):
            return torch.zeros((1, 8, 7))

    policy.action_model = ActionHead()
    policy._stats.update({"dino_calls": 0, "action_calls": 0})
    policy.async_stats = lambda: {"latest_vlm_step": policy._cached_vlm_step}

    output = policy.predict_action([{}], control_step=8)

    assert len(submitted) == 1
    assert output["normalized_actions"].shape == (1, 8, 7)
    assert output["async_stats"]["latest_vlm_step"] == 0
