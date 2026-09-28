# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""MiniCPM-V 4.6 native-video message packing for residual prediction.

Each observation is represented as a one-frame native video block followed by
its explicit episode step, absolute timestamp, chronological frame index, and
view id. This form preserves MiniCPM's video-token path while making each
observation an exact append-only message for the optional hybrid-cache path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image
from transformers.video_utils import VideoMetadata


@dataclass(frozen=True)
class HistoryFrame:
    image: Image.Image
    episode_id: str
    control_step: int
    timestamp_seconds: float
    view_id: str
    frame_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError(f"HistoryFrame.image must be PIL.Image, got {type(self.image)!r}")
        if int(self.control_step) < 0 or int(self.frame_order) < 0:
            raise ValueError("control_step and frame_order must be nonnegative")
        if not torch.isfinite(torch.tensor(float(self.timestamp_seconds))):
            raise ValueError(f"invalid timestamp_seconds={self.timestamp_seconds!r}")
        if not self.view_id:
            raise ValueError("view_id must be explicit and nonempty")


class MiniCPMVideoHistoryAdapter:
    """Build and tokenize identical timestamped MiniCPM video histories."""

    def __init__(
        self,
        processor,
        residual_token: str = "<|dino_residual|>",
        num_residual_tokens: int = 256,
        fps: float = 10.0,
        video_downsample_mode: str = "16x",
    ) -> None:
        if fps <= 0:
            raise ValueError("video fps must be > 0")
        self.processor = processor
        self.residual_token = residual_token
        self.num_residual_tokens = int(num_residual_tokens)
        self.fps = float(fps)
        self.video_downsample_mode = video_downsample_mode

    def task_message(self, instruction: str, episode_id: str) -> dict[str, Any]:
        text = (
            "Task instruction: "
            + str(instruction)
            + f"\nEpisode id: {episode_id}. The following native-video observations are in chronological order."
        )
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def frame_message(self, frame: HistoryFrame) -> dict[str, Any]:
        time_text = (
            f"frame_order={int(frame.frame_order)}; "
            f"control_step={int(frame.control_step)}; "
            f"timestamp={float(frame.timestamp_seconds):.9f}s; "
            f"view_id={frame.view_id}; episode_id={frame.episode_id}"
        )
        return {
            "role": "user",
            "content": [
                {"type": "video", "video": [frame.image]},
                {"type": "text", "text": time_text},
            ],
        }

    def history_messages(
        self,
        instruction: str,
        episode_id: str,
        history: Sequence[HistoryFrame],
        history_mode: str = "full",
    ) -> list[dict[str, Any]]:
        if not history:
            raise ValueError("cannot encode an empty history")
        selected = list(history) if history_mode == "full" else [history[-1]]
        if history_mode not in {"full", "current_only"}:
            raise ValueError(f"unsupported history_mode={history_mode!r}")
        self._validate_history(selected, episode_id)
        return [self.task_message(instruction, episode_id), *(self.frame_message(f) for f in selected)]

    def query_message(self) -> dict[str, Any]:
        query = "Predict the 16x16 DINOv2 patch-feature residual: " + self.residual_token * self.num_residual_tokens
        return {"role": "user", "content": [{"type": "text", "text": query}]}

    def _metadata_for(self, frames: Sequence[HistoryFrame]) -> list[VideoMetadata]:
        result = []
        for frame in frames:
            result.append(
                VideoMetadata(
                    total_num_frames=1,
                    fps=self.fps,
                    duration=1.0 / self.fps,
                    width=frame.image.width,
                    height=frame.image.height,
                    # Every message is a one-frame video. Absolute chronology
                    # is carried by the adjacent timestamp text, while indices
                    # inside this one-frame metadata object remain local.
                    frames_indices=[0],
                )
            )
        return result

    def tokenize(
        self,
        messages: Sequence[dict[str, Any]],
        history: Sequence[HistoryFrame] = (),
        device: torch.device | str | None = None,
        add_generation_prompt: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Use MiniCPM-V's video path and explicitly disable processor sampling."""
        messages = list(messages)
        kwargs: dict[str, Any] = {
            "text_kwargs": {"padding": True},
            "images_kwargs": {
                "use_image_id": False,
                "downsample_mode": self.video_downsample_mode,
            },
            "videos_kwargs": {
                "do_sample_frames": False,
                "video_metadata": self._metadata_for(history) if history else None,
                "downsample_mode": self.video_downsample_mode,
            },
        }
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=kwargs,
        )
        if device is not None:
            encoded = encoded.to(device)
        return dict(encoded)

    def tokenize_history_query(
        self,
        instruction: str,
        episode_id: str,
        history: Sequence[HistoryFrame],
        history_mode: str = "full",
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        messages = self.history_messages(instruction, episode_id, history, history_mode)
        messages.append(self.query_message())
        selected = list(history) if history_mode == "full" else [history[-1]]
        return self.tokenize(messages, selected, device=device)

    def tokenize_history_with_text(
        self,
        instruction: str,
        episode_id: str,
        history: Sequence[HistoryFrame],
        assistant_text: str,
        history_mode: str = "full",
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return separate answer-free prefix and teacher-forced text inputs."""
        if not assistant_text or not assistant_text.strip():
            raise ValueError("assistant_text must be an explicit nonempty label")
        prefix_messages = self.history_messages(instruction, episode_id, history, history_mode)
        selected = list(history) if history_mode == "full" else [history[-1]]
        # The generation header belongs to the conditioning prefix. This lets
        # the trainer mask only assistant answer tokens, including labels from
        # templates that do not expose a Jinja ``generation`` block.
        prefix = self.tokenize(
            prefix_messages, selected, device=device, add_generation_prompt=True
        )
        full_messages = [*prefix_messages, {"role": "assistant", "content": assistant_text}]
        full = self.tokenize(full_messages, selected, device=device)
        return prefix, full

    def _validate_history(self, history: Sequence[HistoryFrame], episode_id: str) -> None:
        prev_step = -1
        prev_order = -1
        prev_timestamp = float("-inf")
        for frame in history:
            if str(frame.episode_id) != str(episode_id):
                raise ValueError(
                    f"history crossed episode boundary: expected {episode_id!r}, got {frame.episode_id!r}"
                )
            if frame.control_step <= prev_step or frame.frame_order <= prev_order:
                raise ValueError("history frames must be strictly increasing by control step and frame order")
            if frame.timestamp_seconds < prev_timestamp:
                raise ValueError("history timestamps must be nondecreasing")
            prev_step, prev_order = frame.control_step, frame.frame_order
            prev_timestamp = float(frame.timestamp_seconds)


def gather_token_hidden_states(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    expected_count: int = 256,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather repeated query-token states correctly with left/right padding."""
    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("hidden_states must be [B,L,H] and input_ids must be [B,L]")
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError(
            f"hidden/input sequence shapes disagree: {tuple(hidden_states.shape[:2])} vs {tuple(input_ids.shape)}"
        )
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    query_mask = input_ids.eq(int(token_id)) & valid
    counts = query_mask.sum(dim=1)
    if not torch.all(counts == expected_count):
        raise ValueError(
            f"expected {expected_count} residual query tokens in every sample; "
            f"found {counts.detach().cpu().tolist()} (token_id={token_id})"
        )
    positions = [torch.nonzero(query_mask[row], as_tuple=False).flatten() for row in range(input_ids.shape[0])]
    return torch.stack([hidden_states[row].index_select(0, pos) for row, pos in enumerate(positions)], dim=0)


def valid_token_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
    """Return each sequence without its left or right padding."""
    if attention_mask is None:
        return [row for row in input_ids]
    return [row[mask.to(dtype=torch.bool)] for row, mask in zip(input_ids, attention_mask)]
