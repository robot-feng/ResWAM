import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.minicpm_asy_temporal_sampler import (
    MiniCPMAsyncTemporalSampler,
    resolve_async_training_alignment,
)
import starVLA.dataloader as dataloader_module
import starVLA.dataloader.lerobot_datasets as lerobot_datasets_module


def _sampler(mode="fixed_step_delay", latency=4, trace=None, trace_max=None, cache_size=0):
    return MiniCPMAsyncTemporalSampler(
        {
            "mode": mode,
            "refresh_interval": 8,
            "fixed_latency_steps": latency if mode == "fixed_step_delay" else None,
            "trace_activation_steps": trace,
            "trace_max_control_step": trace_max,
        },
        cache_size=cache_size,
    )


def _frame(value):
    return Image.new("RGB", (2, 2), color=(value, 0, 0))


def test_fixed_delay_sampler_preserves_current_sample_and_adds_past_vlm_anchor():
    sampler = _sampler()
    current_image = _frame(13)
    action = np.full((8, 7), 13, dtype=np.float32)
    sample = {"image": [current_image], "action": action, "lang": "task"}
    loads = []

    aligned = sampler.align_sample(
        sample,
        control_step=13,
        load_anchor_images=lambda step: loads.append(step) or [_frame(step)],
        cache_key=("episode", 4),
    )

    assert loads == [8]
    assert aligned["image"] is sample["image"]
    assert aligned["action"] is action
    assert aligned["vlm_image"][0].getpixel((0, 0)) == (8, 0, 0)
    assert aligned["vlm_source_step"] == aligned["vlm_anchor_frame"] == 8
    assert aligned["vlm_request_step"] == 8
    assert aligned["vlm_activation_step"] == 12
    assert aligned["vlm_current_step"] == 13
    assert aligned["vlm_age_steps"] == 5
    assert aligned["vlm_delivery_latency_steps"] == 4
    assert aligned["vlm_alignment_mode"] == "fixed_step_delay"


def test_sampler_reuses_anchor_only_within_the_same_episode_key():
    sampler = _sampler(cache_size=32)
    loads = []

    def load(step):
        loads.append(step)
        return [_frame(step)]

    first = sampler.align_sample(
        {"image": [_frame(13)], "action": "a"},
        control_step=13,
        load_anchor_images=load,
        cache_key=("dataset", 1),
    )
    second = sampler.align_sample(
        {"image": [_frame(14)], "action": "b"},
        control_step=14,
        load_anchor_images=load,
        cache_key=("dataset", 2),
    )

    assert loads == [8, 8]
    assert first["vlm_image"][0].getpixel((0, 0)) == (8, 0, 0)
    assert second["vlm_image"][0].getpixel((0, 0)) == (8, 0, 0)


def test_synchronous_boundary_uses_current_frame_without_loading_an_anchor():
    sampler = _sampler(mode="synchronous", latency=None)
    current = [_frame(8)]
    sample = {"image": current, "action": "current-action"}

    aligned = sampler.align_sample(
        sample,
        control_step=8,
        load_anchor_images=lambda step: pytest.fail("same-step source should reuse current image"),
        cache_key=("episode", 1),
    )

    assert aligned["vlm_source_step"] == 8
    assert aligned["vlm_activation_step"] == 8
    assert aligned["vlm_image"][0].getpixel((0, 0)) == (8, 0, 0)


def test_trace_sampler_rejects_samples_past_recorded_coverage():
    sampler = _sampler(
        mode="trace_replay",
        trace={0: 0, 8: 11},
        trace_max=15,
    )
    with pytest.raises(ValueError, match="exceeds alignment trace coverage"):
        sampler.align_sample(
            {"image": [_frame(16)], "action": "a"},
            control_step=16,
            load_anchor_images=lambda step: [_frame(step)],
            cache_key=("episode", 1),
        )


def test_wall_clock_training_requires_a_trace_and_resolves_it_for_offline_sampling(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"async_step_trace": ['
        '{"control_step": 0, "cached_vlm_step": 0}, '
        '{"control_step": 11, "cached_vlm_step": 8}]}'
    )
    cfg = OmegaConf.create(
        {
            "name": "MiniCPMGR00TDualAsy",
            "vlm_refresh_interval": 8,
            "async_alignment": {
                "mode": "wall_clock",
                "training_trace_path": str(trace),
            },
        }
    )

    resolved = resolve_async_training_alignment(cfg)

    assert resolved["runtime_mode"] == "wall_clock"
    assert resolved["mode"] == "trace_replay"
    assert resolved["trace_activation_steps"] == {0: 0, 8: 11}
    assert resolved["trace_max_control_step"] == 11

    cfg.framework = {"async_alignment": {"mode": "wall_clock"}}
    with pytest.raises(ValueError, match="requires framework.async_alignment.training_trace_path"):
        resolve_async_training_alignment(cfg.framework)


def test_trace_replay_training_and_runtime_must_share_the_same_trace(tmp_path):
    trace_a = tmp_path / "runtime.json"
    trace_b = tmp_path / "training.json"
    trace_a.write_text(
        '{"async_step_trace": [{"control_step": 0, "cached_vlm_step": 0}]}'
    )
    trace_b.write_text(
        '{"async_step_trace": [{"control_step": 0, "cached_vlm_step": 0}]}'
    )
    cfg = OmegaConf.create(
        {
            "name": "MiniCPMGR00TDualAsy",
            "vlm_refresh_interval": 8,
            "async_alignment": {
                "mode": "trace_replay",
                "trace_path": str(trace_a),
                "training_trace_path": str(trace_b),
            },
        }
    )
    with pytest.raises(ValueError, match="must use the same activation trace"):
        resolve_async_training_alignment(cfg)


def test_general_sampler_rejects_nonunit_execution_horizon_until_boundary_sampling_exists():
    cfg = OmegaConf.create(
        {
            "name": "MiniCPMGR00TDualAsy",
            "execution_horizon": 4,
            "vlm_refresh_interval": 8,
            "async_alignment": {
                "mode": "fixed_step_delay",
                "fixed_latency_steps": 4,
            },
        }
    )
    with pytest.raises(ValueError, match="currently aligns K=1"):
        resolve_async_training_alignment(cfg)


def test_lerobot_mixture_returns_anchor_metadata_without_changing_current_action():
    sampler = _sampler()

    class FakeSingleDataset:
        dataset_name = "fake-libero"
        lerobot_info_meta = {"total_videos": 0}
        modality_keys = {"video": ["video.front"]}

        def get_step_data(self, trajectory_id, step):
            return {"trajectory_id": trajectory_id, "step": step}

        def transforms(self, raw):
            return raw

        def _pack_sample(self, raw):
            return {
                "image": [_frame(raw["step"])],
                "action": np.full((8, 7), raw["step"], dtype=np.float32),
                "lang": f"task-{raw['trajectory_id']}",
            }

        def _apply_async_temporal_alignment(self, sample, trajectory_id, control_step):
            return sampler.align_sample(
                sample,
                control_step=control_step,
                load_anchor_images=lambda step: self._pack_sample(
                    self.transforms(self.get_step_data(trajectory_id, step))
                )["image"],
                cache_key=("fake-libero", trajectory_id),
            )

    mixture = object.__new__(LeRobotMixtureDataset)
    mixture._getitem_count = 0
    mixture.datasets = [FakeSingleDataset()]
    mixture.sample_step = lambda index: (mixture.datasets[0], 42, 13)

    sample = mixture.__getitem__(0)

    assert sample["lang"] == "task-42"
    assert sample["action"].shape == (8, 7)
    assert np.all(sample["action"] == 13)
    assert sample["image"][0].getpixel((0, 0)) == (13, 0, 0)
    assert sample["vlm_image"][0].getpixel((0, 0)) == (8, 0, 0)
    assert sample["vlm_source_step"] == 8


def test_get_vla_dataset_attaches_the_temporal_sampler_only_when_configured(monkeypatch):
    class FakeSingleDataset:
        def __init__(self):
            self.sampler = None

        def set_async_temporal_sampler(self, sampler):
            self.sampler = sampler

    single = FakeSingleDataset()
    mixture_capture = {}
    monkeypatch.setattr(
        lerobot_datasets_module,
        "DATASET_NAMED_MIXTURES",
        {"fake": [("demo", 1.0, "libero_franka")]},
    )
    monkeypatch.setattr(
        lerobot_datasets_module,
        "make_LeRobotSingleDataset",
        lambda *args, **kwargs: single,
    )

    def capture_mixture(data_mixture, **kwargs):
        mixture_capture["data_mixture"] = data_mixture
        mixture_capture["kwargs"] = kwargs
        return "mixture"

    monkeypatch.setattr(
        lerobot_datasets_module, "LeRobotMixtureDataset", capture_mixture
    )
    result = lerobot_datasets_module.get_vla_dataset(
        OmegaConf.create({"data_root_dir": "/unused", "data_mix": "fake"}),
        async_alignment={
            "mode": "fixed_step_delay",
            "refresh_interval": 8,
            "fixed_latency_steps": 4,
        },
    )

    assert result == "mixture"
    assert isinstance(single.sampler, MiniCPMAsyncTemporalSampler)
    assert single.sampler.mode == "fixed_step_delay"
    assert mixture_capture["kwargs"]["data_cfg"].data_mix == "fake"
    assert "async_alignment" not in mixture_capture["kwargs"]


def test_build_dataloader_resolves_wall_clock_training_trace_for_lerobot(monkeypatch, tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"async_step_trace": ['
        '{"control_step": 0, "cached_vlm_step": 0}, '
        '{"control_step": 11, "cached_vlm_step": 8}]}'
    )
    captured = {}

    class FakeVlaDataset:
        def save_dataset_statistics(self, path):
            captured["statistics_path"] = path

    def fake_get_vla_dataset(**kwargs):
        captured.update(kwargs)
        return FakeVlaDataset()

    monkeypatch.setattr(
        lerobot_datasets_module, "get_vla_dataset", fake_get_vla_dataset
    )
    monkeypatch.setattr(
        dataloader_module,
        "DataLoader",
        lambda dataset, **kwargs: (dataset, kwargs),
    )
    monkeypatch.setattr(dataloader_module.logger, "info", lambda *args, **kwargs: None)
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "framework": {
                "name": "MiniCPMGR00TDualAsy",
                "vlm_refresh_interval": 8,
                "async_alignment": {
                    "mode": "wall_clock",
                    "training_trace_path": str(trace),
                },
            },
            "datasets": {
                "vla_data": {
                    "dataset_py": "lerobot_datasets",
                    "per_device_batch_size": 1,
                    "num_workers": 0,
                }
            },
        }
    )

    dataset, loader_kwargs = dataloader_module.build_dataloader(
        cfg, dataset_py="lerobot_datasets"
    )

    assert isinstance(dataset, FakeVlaDataset)
    assert loader_kwargs["num_workers"] == 0
    assert captured["async_alignment"]["runtime_mode"] == "wall_clock"
    assert captured["async_alignment"]["mode"] == "trace_replay"
    assert captured["async_alignment"]["trace_activation_steps"] == {0: 0, 8: 11}


def test_build_dataloader_fails_closed_without_training_trace(tmp_path):
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "MiniCPMGR00TDualAsy",
                "async_alignment": {"mode": "wall_clock"},
            },
            "datasets": {"vla_data": {"dataset_py": "lerobot_datasets"}},
        }
    )
    with pytest.raises(ValueError, match="training_trace_path"):
        dataloader_module.build_dataloader(cfg, dataset_py="lerobot_datasets")


def test_dualasy_rejects_non_lerobot_loader_until_it_can_attach_temporal_anchors():
    cfg = OmegaConf.create({"framework": {"name": "MiniCPMGR00TDualAsy"}})
    with pytest.raises(ValueError, match="requires datasets.vla_data.dataset_py=lerobot_datasets"):
        dataloader_module.build_dataloader(cfg, dataset_py="vlm_datasets")
