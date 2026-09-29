import json
import queue
import threading

import pytest
import torch

from starVLA.model.framework.VLM4A.MiniCPMGR00TDualAsy import MiniCPMGR00TDualAsy
from starVLA.model.framework.VLM4A.minicpm_dual_asy_alignment import (
    fixed_activation_steps,
    load_trace_activation_steps,
    load_trace_max_control_step,
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


def test_fixed_schedule_activates_only_at_action_call_boundaries():
    events = fixed_activation_steps(2, 1, 12, execution_horizon=4)
    assert events == {0: 0, 2: 4, 4: 8, 6: 8, 8: 12, 10: 12, 12: 16}
    assert source_step_at(
        4,
        mode="fixed_step_delay",
        refresh_interval=2,
        fixed_latency_steps=1,
        execution_horizon=4,
    ) == 2
    assert source_step_at(
        8,
        mode="fixed_step_delay",
        refresh_interval=2,
        fixed_latency_steps=1,
        execution_horizon=4,
    ) == 6


@pytest.mark.parametrize(
    ("latency", "expected_call_sources"),
    [
        (0, [0, 4, 8, 12]),
        (1, [0, 2, 6, 10]),
        (2, [0, 2, 6, 10]),
        (4, [0, 0, 2, 4]),
    ],
)
def test_k4_m2_controlled_latency_grid(latency, expected_call_sources):
    assert [
        source_step_at(
            step,
            mode="fixed_step_delay",
            refresh_interval=2,
            fixed_latency_steps=latency,
            execution_horizon=4,
        )
        for step in (0, 4, 8, 12)
    ] == expected_call_sources


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


@pytest.mark.parametrize("latency", [0, 1, 2, 4])
def test_fixed_runtime_source_matches_training_source_schedule(latency):
    policy = _stub_async_policy("fixed_step_delay")
    policy.vlm_refresh_interval = 8
    policy.execution_horizon = 1
    policy.fixed_latency_steps = latency
    for step in range(40):
        assert policy._scheduled_source_step(step) == source_step_at(
            step,
            mode="fixed_step_delay",
            refresh_interval=8,
            fixed_latency_steps=latency,
        )


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
    assert load_trace_max_control_step(trace) == 19


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


def test_trace_replay_rejects_non_fifo_activation_order():
    with pytest.raises(ValueError, match="monotonic"):
        source_step_at(
            20,
            mode="trace_replay",
            refresh_interval=8,
            trace_activation_steps={0: 0, 8: 18, 16: 17},
        )


def test_trace_loader_skips_empty_non_async_baseline_trace(tmp_path):
    trace = tmp_path / "comparison.json"
    trace.write_text(
        json.dumps(
            {
                "models": {
                    "MiniCPMGR00TDual": {"simulation": {"async_step_trace": []}},
                    "MiniCPMGR00TDualAsy": {
                        "simulation": {
                            "async_step_trace": [
                                {"control_step": 0, "cached_vlm_step": 0},
                                {"control_step": 11, "cached_vlm_step": 8},
                            ]
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_trace_activation_steps(trace) == {0: 0, 8: 11}
    assert load_trace_max_control_step(trace) == 11


def _stub_async_policy(mode):
    policy = object.__new__(MiniCPMGR00TDualAsy)
    torch.nn.Module.__init__(policy)
    policy.async_alignment_mode = mode
    policy._async_generation = 1
    policy._control_step = 12
    policy._control_step_origin = None
    policy._runtime_lock = threading.RLock()
    policy._cached_vlm_hidden = None
    policy._cached_instruction = None
    policy._observed_instruction_key = None
    policy._cached_vlm_step = None
    policy._cached_ready_timestamp = None
    policy._last_refresh_source_step = None
    policy._last_observed_step = None
    policy._last_policy_call_step = None
    policy.execution_horizon = 1
    policy._completed_snapshots = {}
    policy._refresh_events = []
    policy._refresh_events_by_key = {}
    policy._activation_events = []
    policy._result_queue = queue.Queue()
    policy._request_queue = queue.Queue()
    policy._stats = {
        "vlm_submitted": 0,
        "vlm_completed": 0,
        "vlm_dropped_stale": 0,
        "dino_calls": 0,
        "action_calls": 0,
        "vlm_seconds": [],
    }
    return policy


def test_observe_path_submits_vlm_on_control_steps_between_action_calls():
    policy = _stub_async_policy("wall_clock")
    policy.vlm_refresh_interval = 2
    policy.vlm_update_interval = 2
    policy.execution_horizon = 4
    policy._control_step_origin = 0
    policy._control_step = 2
    policy._last_refresh_source_step = 0
    policy.align_model_input = lambda examples: (["front"], None, ["task"], None)
    submitted = []
    policy._submit_vlm_refresh = lambda images, instructions, step, source_timestamp=None: submitted.append(step)

    result = policy.observe_for_async_refresh(
        {"image": ["frame-2"], "lang": "task"}, control_step=2
    )

    assert submitted == [2]
    assert result["observed_control_step"] == 2
    assert policy._last_observed_step == 2


def test_wall_clock_observation_is_not_action_active_until_next_policy_call():
    policy = _stub_async_policy("wall_clock")
    policy.vlm_refresh_interval = 2
    policy.vlm_update_interval = 2
    policy.execution_horizon = 4
    policy._control_step_origin = 0
    policy._control_step = 2
    policy._last_refresh_source_step = 0
    policy._cached_vlm_hidden = torch.zeros((1, 2, 3))
    policy._cached_vlm_step = 0
    policy._cached_instruction = ("task",)
    policy._observed_instruction_key = ("task",)
    policy.align_model_input = lambda examples: (["front"], None, ["task"], None)

    def complete_refresh(images, instructions, step, source_timestamp=None):
        timestamp = 10.0 if source_timestamp is None else source_timestamp
        policy._new_refresh_event(step, timestamp, timestamp)
        policy._stats["vlm_submitted"] += 1
        policy._result_queue.put(
            {
                "generation": policy._async_generation,
                "source_step": step,
                "source_timestamp": timestamp,
                "instructions": tuple(instructions),
                "hidden": torch.ones((1, 2, 3)),
                "compute_seconds": 0.1,
                "request_timestamp": timestamp,
                "ready_timestamp": timestamp + 0.1,
                "error": None,
            }
        )

    policy._submit_vlm_refresh = complete_refresh

    policy.observe_for_async_refresh(
        {"image": ["frame-2"], "lang": "task"}, control_step=2
    )

    assert policy._cached_vlm_step == 0
    assert 2 in policy._completed_snapshots
    assert policy._refresh_events_by_key[(1, 2)]["ready_step"] == 2

    policy._poll_vlm_results(control_step=4)

    assert policy._cached_vlm_step == 2
    assert policy._activation_events[-1]["activation_step"] == 4


def test_policy_inference_rejects_non_call_boundary_for_configured_k():
    policy = _stub_async_policy("wall_clock")
    policy.execution_horizon = 4
    policy.vlm_refresh_interval = 8
    policy.vlm_update_interval = 8
    policy._control_step_origin = 0
    policy.align_model_input = lambda examples: (["front"], ["wrist"], ["task"], None)

    with pytest.raises(ValueError, match="policy-call boundaries"):
        policy._predict_action_impl([{"image": ["frame"], "lang": "task"}], control_step=2)


def test_k_greater_than_one_requires_all_intermediate_observations():
    policy = _stub_async_policy("wall_clock")
    policy.execution_horizon = 4
    policy.vlm_refresh_interval = 8
    policy.vlm_update_interval = 8
    policy._control_step_origin = 0
    policy._last_policy_call_step = 0
    policy._last_observed_step = 2
    policy.align_model_input = lambda examples: (["front"], ["wrist"], ["task"], None)

    with pytest.raises(ValueError, match="every intervening control observation"):
        policy._predict_action_impl([{"image": ["frame"], "lang": "task"}], control_step=4)


def test_instruction_change_invalidates_pending_semantic_generation():
    policy = _stub_async_policy("wall_clock")
    policy.vlm_refresh_interval = 8
    policy.vlm_update_interval = 8
    policy._control_step_origin = 0
    policy._control_step = 4
    policy._observed_instruction_key = ("old task",)
    policy.align_model_input = lambda examples: (["front"], None, ["new task"], None)
    policy._submit_vlm_refresh = lambda *args, **kwargs: None

    policy.observe_for_async_refresh(
        {"image": ["frame-4"], "lang": "new task"}, control_step=4
    )

    assert policy._async_generation == 2
    assert policy._observed_instruction_key == ("new task",)
    assert policy._last_refresh_source_step == 0


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
    assert policy._activation_events[0]["ready_step"] == 11
    assert policy._activation_events[0]["request_timestamp"] == 10.0
    assert policy._activation_events[0]["ready_timestamp"] == 10.25


def test_synchronous_refresh_is_ready_and_active_on_its_request_step():
    policy = _stub_async_policy("synchronous")
    policy._control_step = 1
    policy._encode_vlm = lambda images, instructions: torch.ones((1, 2, 3))

    policy._run_synchronous_refresh(
        ["frame"], ["task"], control_step=0, source_timestamp=9.0
    )

    event = policy._activation_events[0]
    assert policy._cached_vlm_step == 0
    assert event["request_step"] == 0
    assert event["ready_step"] == 0
    assert event["activation_step"] == 0
    assert event["source_timestamp"] == 9.0


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

    policy._poll_vlm_results(control_step=12)

    assert policy._cached_vlm_step == 8
    assert policy._activation_events[0]["ready_step"] == 12
    assert policy._activation_events[0]["activation_step"] == 12
    assert policy._activation_events[0]["activation_timestamp"] >= 10.25


def test_reset_invalidates_cached_and_inflight_generation_results():
    policy = _stub_async_policy("wall_clock")
    policy._cached_vlm_hidden = torch.ones((1, 2, 3))
    policy._cached_instruction = ("old task",)
    policy._cached_vlm_step = 8
    old_generation = policy._async_generation

    policy.reset_async_cache()
    stale_result = {
        "generation": old_generation,
        "source_step": 16,
        "instructions": ("old task",),
        "hidden": torch.full((1, 2, 3), 2.0),
        "compute_seconds": 0.3,
        "request_timestamp": 10.0,
        "ready_timestamp": 10.3,
        "error": None,
    }

    assert policy._accept_vlm_result(stale_result, ready_control_step=0) is None
    assert policy._cached_vlm_hidden is None
    assert policy._cached_instruction is None
    assert policy._cached_vlm_step is None
    assert policy._async_generation == old_generation + 1


def test_vlm_worker_does_not_wait_on_control_lock(monkeypatch):
    policy = _stub_async_policy("fixed_step_delay")
    policy._worker_stream = None
    policy._async_generation = 1
    policy._stats["vlm_dropped_stale"] = 0
    policy._encode_vlm = lambda images, instructions: torch.ones((1, 2, 3))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    worker = threading.Thread(target=policy._vlm_worker_loop, daemon=True)
    result = None
    policy._runtime_lock.acquire()
    try:
        worker.start()
        policy._request_queue.put((1, 0, [], ("task",), 1.0, 1.0))
        try:
            result = policy._result_queue.get(timeout=1.0)
        except queue.Empty:
            pass
    finally:
        policy._runtime_lock.release()
        policy._request_queue.put(None)
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert result is not None
    assert result["source_step"] == 0
    assert result["error"] is None


def test_control_step_origin_resets_with_instruction_epoch():
    policy = _stub_async_policy("wall_clock")
    assert policy._resolve_control_step(20) == 0
    assert policy._resolve_control_step(21) == 1
    policy.reset_async_cache()
    assert policy._resolve_control_step(37) == 0
    with pytest.raises(ValueError, match="moved backwards"):
        policy._resolve_control_step(36)


def test_optional_control_step_api_keeps_a_consistent_local_epoch():
    policy = _stub_async_policy("wall_clock")
    policy._control_step = 3
    assert policy._resolve_control_step(None) == 3
    assert policy._resolve_control_step(4) == 4


def test_instruction_change_resets_context_and_starts_new_local_timeline():
    policy = _stub_async_policy("wall_clock")
    policy._control_step = 11
    policy._control_step_origin = 0
    policy._cached_instruction = ("old task",)
    policy._cached_vlm_hidden = torch.ones((1, 2, 3))
    policy._cached_vlm_step = 8
    policy.vlm_update_interval = 8
    policy.align_model_input = lambda examples: ([], None, ["new task"], None)

    def submit(images, instructions, control_step, source_timestamp=None):
        policy._result_queue.put(
            {
                "generation": policy._async_generation,
                "source_step": control_step,
                "source_timestamp": source_timestamp,
                "instructions": tuple(instructions),
                "hidden": torch.full((1, 2, 3), 2.0),
                "compute_seconds": 0.1,
                "request_timestamp": source_timestamp,
                "ready_timestamp": source_timestamp + 0.1,
                "error": None,
            }
        )

    policy._submit_vlm_refresh = submit
    policy._condition_from_vlm_hidden = lambda *args: (torch.ones((1, 2, 3)), None)

    class ActionHead(torch.nn.Module):
        def predict_action(self, condition, state):
            return torch.zeros((1, 8, 7))

    policy.action_model = ActionHead()
    policy._stats.update({"dino_calls": 0, "action_calls": 0})

    output = policy.predict_action([{}], control_step=10)

    assert output["normalized_actions"].shape == (1, 8, 7)
    assert policy._async_generation == 2
    assert policy._control_step_origin == 10
    assert policy._stats["last_control_step"] == 0
    assert policy._cached_instruction == ("new task",)
    assert policy._cached_vlm_step == 0


def test_policy_instances_keep_separate_semantic_caches():
    first = _stub_async_policy("wall_clock")
    second = _stub_async_policy("wall_clock")
    first._cached_vlm_hidden = torch.ones((1, 2, 3))
    first._cached_vlm_step = 4
    second._cached_vlm_hidden = torch.full((1, 2, 3), 2.0)
    second._cached_vlm_step = 8

    first.reset_async_cache()

    assert first._cached_vlm_hidden is None
    assert first._cached_vlm_step is None
    assert torch.all(second._cached_vlm_hidden == 2.0)
    assert second._cached_vlm_step == 8


def test_wall_clock_action_uses_old_cache_without_waiting_for_refresh():
    policy = _stub_async_policy("wall_clock")
    policy._control_step = 9
    policy._cached_instruction = ("task",)
    policy._cached_vlm_hidden = torch.ones((1, 2, 3))
    policy._cached_vlm_step = 0
    policy.vlm_update_interval = 8
    submitted = []
    policy.align_model_input = lambda examples: ([], None, ["task"], None)
    policy._submit_vlm_refresh = lambda *args, **kwargs: submitted.append((args, kwargs))
    policy._poll_vlm_results = lambda control_step=None: None
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
    assert policy._stats["last_action_compute_seconds"] is not None
    assert policy._stats["last_condition_compute_seconds"] is not None


def test_wall_clock_activation_discards_consumed_hidden_snapshots():
    policy = _stub_async_policy("wall_clock")
    policy._cached_vlm_hidden = torch.zeros((1, 2, 3))
    policy._cached_vlm_step = 0
    policy._completed_snapshots = {
        source_step: {
            "generation": 1,
            "source_step": source_step,
            "source_timestamp": float(source_step),
            "instructions": ("task",),
            "hidden": torch.full((1, 2, 3), float(source_step)),
            "compute_seconds": 0.1,
            "request_timestamp": float(source_step),
            "ready_timestamp": float(source_step) + 0.1,
            "ready_step": source_step + 1,
            "error": None,
        }
        for source_step in (0, 8)
    }

    policy._poll_vlm_results(control_step=12)

    assert policy._cached_vlm_step == 8
    assert torch.all(policy._cached_vlm_hidden == 8.0)
    assert policy._completed_snapshots == {}
    assert policy._activation_events[-1]["activation_step"] == 12


def test_controlled_activation_prunes_old_but_keeps_future_ready_snapshots():
    policy = _stub_async_policy("trace_replay")
    policy._completed_snapshots = {step: {"hidden": torch.zeros(1)} for step in (0, 8, 16)}

    policy._discard_completed_snapshots_through(8)

    assert set(policy._completed_snapshots) == {16}


def test_closing_worker_releases_unactivated_hidden_snapshots():
    policy = _stub_async_policy("wall_clock")
    policy._completed_snapshots = {8: {"hidden": torch.ones(1)}}

    class StoppedWorker:
        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    policy._worker = StoppedWorker()
    policy.close_async_worker()

    assert policy._completed_snapshots == {}
    assert policy._worker is None


def test_trace_replay_allows_a_refresh_that_was_never_action_visible():
    policy = _stub_async_policy("trace_replay")
    policy._control_step = 8
    policy._control_step_origin = 0
    policy._cached_instruction = ("task",)
    policy._cached_vlm_hidden = torch.ones((1, 2, 3))
    policy._cached_vlm_step = 0
    policy._completed_snapshots[0] = {
        "source_step": 0,
        "instructions": ("task",),
        "hidden": policy._cached_vlm_hidden,
        "request_timestamp": 1.0,
        "ready_timestamp": 1.1,
        "compute_seconds": 0.1,
        "ready_step": 0,
    }
    policy.vlm_refresh_interval = 8
    policy.vlm_update_interval = 8
    policy.fixed_latency_steps = None
    policy.trace_activation_steps = {0: 0, 16: 19}
    policy.trace_max_control_step = 20
    submitted = []
    policy.align_model_input = lambda examples: ([], None, ["task"], None)
    policy._submit_vlm_refresh = lambda *args, **kwargs: submitted.append(args[2])
    policy._condition_from_vlm_hidden = lambda *args: (torch.ones((1, 2, 3)), None)

    class ActionHead(torch.nn.Module):
        def predict_action(self, condition, state):
            return torch.zeros((1, 8, 7))

    policy.action_model = ActionHead()
    policy._stats.update({"dino_calls": 0, "action_calls": 0})

    output = policy.predict_action([{}], control_step=8)

    assert submitted == [8]
    assert policy._cached_vlm_step == 0
    assert output["normalized_actions"].shape == (1, 8, 7)
