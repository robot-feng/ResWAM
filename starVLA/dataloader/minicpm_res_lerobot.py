# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Read-only LeRobot adapter for successful-terminal MiniCPM goal samples.

Terminal supervision is loaded only from an explicit JSONL manifest. A recorded
last frame is never assumed to be a successful goal. Current-only mode decodes
only the selected observation and the explicitly annotated terminal frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.VLM4A.minicpm_video_history import HistoryFrame


def _load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"annotation manifest does not exist: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def load_success_terminals(path: str | Path) -> dict[str, int]:
    """Map explicit successful ``episode_id`` records to terminal control step."""
    result: dict[str, int] = {}
    for row in _load_jsonl(path):
        if row.get("is_success") is not True:
            continue
        if "episode_id" not in row or "terminal_step" not in row:
            raise ValueError("success rows require episode_id and terminal_step")
        episode_id = str(row["episode_id"])
        terminal_step = int(row["terminal_step"])
        if terminal_step < 0:
            raise ValueError(f"terminal_step must be nonnegative for episode {episode_id}")
        if episode_id in result and result[episode_id] != terminal_step:
            raise ValueError(f"conflicting successful terminal steps for episode {episode_id}")
        result[episode_id] = terminal_step
    if not result:
        raise ValueError(f"no successful-terminal annotations found in {path}")
    return result


def load_assistant_labels(path: str | Path | None) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for row in _load_jsonl(path):
        for key in ("episode_id", "control_step", "text"):
            if key not in row:
                raise ValueError(f"assistant label rows require {key}")
        text = str(row["text"]).strip()
        if not text:
            raise ValueError("assistant text labels must be nonempty")
        sample_key = (str(row["episode_id"]), int(row["control_step"]))
        if sample_key in result:
            raise ValueError(f"duplicate assistant text label for {sample_key}")
        result[sample_key] = text
    return result


def _to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.ndim == 4:
        if array.shape[0] == 0:
            raise ValueError("dataset returned an empty video frame array")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected HWC image data, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _first_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            found = _first_string(item)
            if found:
                return found
    return None


class MiniCPMResidualLeRobotDataset(Dataset):
    """Adapt one LeRobot dataset to explicit terminal supervision."""

    def __init__(
        self,
        data_root_dir: str | Path,
        dataset_name: str,
        robot_type: str,
        success_terminal_manifest: str | Path,
        assistant_label_manifest: str | Path | None = None,
        camera_key: str = "video.primary_image",
        video_backend: str = "torchvision_av",
        data_cfg: dict[str, Any] | None = None,
        max_history_frames: int | None = None,
        history_mode: str = "full",
        sample_stride: int = 1,
        episode_ids: Sequence[str | int] | None = None,
        control_steps: Sequence[int] | None = None,
    ) -> None:
        self.dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            dataset_name,
            robot_type,
            delete_pause_frame=False,
            data_cfg=data_cfg or {"video_backend": video_backend, "include_state": False},
        )
        if camera_key not in self.dataset.modality_keys.get("video", []):
            raise KeyError(
                f"camera_key={camera_key!r} not present in dataset video modalities "
                f"{self.dataset.modality_keys.get('video', [])}"
            )
        self.camera_key = camera_key
        if history_mode not in {"full", "current_only"}:
            raise ValueError("history_mode must be 'full' or 'current_only'")
        self.history_mode = history_mode
        self.success_terminals = load_success_terminals(success_terminal_manifest)
        self.assistant_labels = load_assistant_labels(assistant_label_manifest)
        self.max_history_frames = None if max_history_frames is None else int(max_history_frames)
        if self.max_history_frames is not None and self.max_history_frames < 1:
            raise ValueError("max_history_frames must be positive when provided")
        self.sample_stride = int(sample_stride)
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        all_steps = [tuple(item) for item in self.dataset.all_steps]
        self._episode_cache_id: str | None = None
        self._episode_cache: dict[str, Any] | None = None
        self._current_only_cache_id: str | None = None
        self._current_only_cache: dict[str, Any] | None = None
        known_episode_ids = {str(trajectory_id) for trajectory_id, _ in all_steps}
        unknown = set(self.success_terminals) - known_episode_ids
        if unknown:
            raise ValueError(
                "terminal manifest episode_id values must match LeRobot trajectory IDs; "
                f"unknown examples: {sorted(unknown)[:8]}"
            )
        # Only explicitly successful episodes have terminal residual targets.
        self._steps = [
            item
            for item in all_steps
            if str(item[0]) in self.success_terminals
            and int(item[1]) <= self.success_terminals[str(item[0])]
            and int(item[1]) % self.sample_stride == 0
        ]
        self.excluded_episodes = len(known_episode_ids - set(self.success_terminals))
        if episode_ids is not None:
            selected_ids = {str(episode_id) for episode_id in episode_ids}
            unknown_ids = selected_ids - known_episode_ids
            if unknown_ids:
                raise ValueError(f"unknown LeRobot episode IDs requested: {sorted(unknown_ids)[:8]}")
            unlabelled_ids = selected_ids - set(self.success_terminals)
            if unlabelled_ids:
                raise ValueError(
                    "requested episodes lack explicit successful-terminal labels: "
                    f"{sorted(unlabelled_ids)[:8]}"
                )
            self._steps = [item for item in self._steps if str(item[0]) in selected_ids]
        if control_steps is not None:
            selected_steps = {int(step) for step in control_steps}
            if not selected_steps or min(selected_steps) < 0:
                raise ValueError("control_steps must contain nonnegative step indices")
            self._steps = [item for item in self._steps if int(item[1]) in selected_steps]
        if not self._steps:
            raise ValueError(
                "no LeRobot steps match the explicit success labels and requested episode/control-step filters"
            )

    def __len__(self) -> int:
        return len(self._steps)

    def _read_step(self, trajectory_id: int, step: int) -> dict[str, Any]:
        raw = self.dataset.get_step_data(trajectory_id, step)
        # get_step_data populates curr_traj_data but (in this starVLA revision)
        # does not set curr_traj_id, so set the cache key locally to avoid
        # rereading the parquet file for every frame in the same episode.
        self.dataset.curr_traj_id = trajectory_id
        return raw

    def _load_episode(self, trajectory_id: int) -> dict[str, Any]:
        episode_id = str(trajectory_id)
        if self._episode_cache_id == episode_id and self._episode_cache is not None:
            return self._episode_cache
        trajectory_index = self.dataset.get_trajectory_index(trajectory_id)
        episode_length = int(self.dataset.trajectory_lengths[trajectory_index])
        if episode_id not in self.success_terminals:
            raise ValueError(
                f"episode {episode_id} has no successful terminal annotation; "
                "provide it in the success-terminal JSONL manifest"
            )
        terminal_step = self.success_terminals[episode_id]
        if terminal_step >= episode_length:
            raise ValueError(
                f"terminal step {terminal_step} is outside episode {episode_id} length {episode_length}"
            )
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        instruction: str | None = None
        for step in range(episode_length):
            raw = self._read_step(trajectory_id, step)
            frames.append(_to_pil(raw[self.camera_key]))
            if self.dataset.curr_traj_data is None or "timestamp" not in self.dataset.curr_traj_data:
                raise ValueError(f"episode {episode_id} has no raw timestamp column")
            timestamps.append(float(self.dataset.curr_traj_data["timestamp"].iloc[step]))
            if instruction is None:
                for key in self.dataset.modality_keys.get("language", []):
                    instruction = _first_string(raw.get(key))
                    if instruction:
                        break
        if not instruction:
            raise ValueError(f"episode {episode_id} has no task instruction in language modality")
        self._episode_cache_id = episode_id
        self._episode_cache = {
            "episode_id": episode_id,
            "frames": frames,
            "timestamps": timestamps,
            "instruction": instruction,
            "terminal_step": terminal_step,
        }
        return self._episode_cache

    def _load_current_only_step(self, trajectory_id: int, control_step: int) -> dict[str, Any]:
        """Read only the selected current frame and its annotated terminal frame."""
        episode_id = str(trajectory_id)
        trajectory_index = self.dataset.get_trajectory_index(trajectory_id)
        episode_length = int(self.dataset.trajectory_lengths[trajectory_index])
        if episode_id not in self.success_terminals:
            raise ValueError(
                f"episode {episode_id} has no successful terminal annotation; "
                "provide it in the success-terminal JSONL manifest"
            )
        terminal_step = self.success_terminals[episode_id]
        if terminal_step >= episode_length:
            raise ValueError(
                f"terminal step {terminal_step} is outside episode {episode_id} length {episode_length}"
            )

        if self._current_only_cache_id != episode_id or self._current_only_cache is None:
            terminal_raw = self._read_step(trajectory_id, terminal_step)
            trajectory_data = self.dataset.curr_traj_data
            if trajectory_data is None or "timestamp" not in trajectory_data:
                raise ValueError(f"episode {episode_id} has no raw timestamp column")
            instruction = None
            for key in self.dataset.modality_keys.get("language", []):
                instruction = _first_string(terminal_raw.get(key))
                if instruction:
                    break
            if not instruction:
                raise ValueError(f"episode {episode_id} has no task instruction in language modality")
            self._current_only_cache_id = episode_id
            self._current_only_cache = {
                "terminal_image": _to_pil(terminal_raw[self.camera_key]),
                "instruction": instruction,
                "timestamps": trajectory_data["timestamp"].to_numpy(copy=True),
                "terminal_step": terminal_step,
            }

        cache = self._current_only_cache
        if control_step == terminal_step:
            current_image = cache["terminal_image"]
        else:
            current_raw = self._read_step(trajectory_id, control_step)
            current_image = _to_pil(current_raw[self.camera_key])
        return {
            "episode_id": episode_id,
            "current_image": current_image,
            "terminal_image": cache["terminal_image"],
            "instruction": cache["instruction"],
            "timestamps": cache["timestamps"],
            "terminal_step": cache["terminal_step"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id, control_step = self._steps[index]
        if self.history_mode == "current_only":
            episode = self._load_current_only_step(trajectory_id, int(control_step))
            timestamp = float(episode["timestamps"][control_step])
            current_frame = HistoryFrame(
                image=episode["current_image"],
                episode_id=episode["episode_id"],
                control_step=int(control_step),
                timestamp_seconds=timestamp,
                view_id=self.camera_key,
                frame_order=int(control_step),
            )
            history = [current_frame]
            current_image = episode["current_image"]
            terminal_image = episode["terminal_image"]
        else:
            episode = self._load_episode(trajectory_id)
            history_start = 0
            if self.max_history_frames is not None:
                # This option is a deliberate context-window ablation and must be
                # reported as such; the default remains every frame so far.
                history_start = max(0, control_step + 1 - self.max_history_frames)
            history = [
                HistoryFrame(
                    image=episode["frames"][step],
                    episode_id=episode["episode_id"],
                    control_step=step,
                    timestamp_seconds=episode["timestamps"][step],
                    view_id=self.camera_key,
                    frame_order=step,
                )
                for step in range(history_start, control_step + 1)
            ]
            timestamp = float(episode["timestamps"][control_step])
            current_image = episode["frames"][control_step]
            terminal_image = episode["frames"][episode["terminal_step"]]
        sample = {
            "episode_id": episode["episode_id"],
            "control_step": int(control_step),
            "timestamp_seconds": timestamp,
            "history": history,
            "current_image": current_image,
            "terminal_image": terminal_image,
            "lang": episode["instruction"],
            "goal_target_valid": True,
            "terminal_step": int(episode["terminal_step"]),
            "view_id": self.camera_key,
        }
        assistant_text = self.assistant_labels.get((episode["episode_id"], int(control_step)))
        if assistant_text is not None:
            sample["assistant_text"] = assistant_text
        return sample
