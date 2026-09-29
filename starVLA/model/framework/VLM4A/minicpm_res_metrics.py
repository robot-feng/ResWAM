# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Comparable goal-space metrics for absolute and residual DINO predictions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def goal_space_metrics(
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    current_features: torch.Tensor,
) -> dict[str, float]:
    """Score a predictor after converting it to the shared goal-feature space.

    Inputs are ``[batch, patches, channels]`` tensors. Both absolute-goal and
    residual heads are converted to the same ``z_goal`` before comparison.
    Residual cosine and norm errors are included as additional diagnostics.
    """
    predicted_residual = predicted_residual.float()
    target_residual = target_residual.float()
    current_features = current_features.float()
    if not (
        predicted_residual.shape == target_residual.shape == current_features.shape
        and predicted_residual.ndim == 3
    ):
        raise ValueError(
            "goal-space metrics require matching [batch, patches, channels] tensors; "
            f"got prediction={tuple(predicted_residual.shape)}, "
            f"target={tuple(target_residual.shape)}, current={tuple(current_features.shape)}"
        )

    predicted_goal = current_features + predicted_residual
    target_goal = current_features + target_residual
    predicted_flat = predicted_goal.flatten(start_dim=1)
    target_flat = target_goal.flatten(start_dim=1)
    residual_pred_flat = predicted_residual.flatten(start_dim=1)
    residual_target_flat = target_residual.flatten(start_dim=1)
    predicted_norm = torch.linalg.vector_norm(residual_pred_flat, dim=1)
    target_norm = torch.linalg.vector_norm(residual_target_flat, dim=1)
    norm_abs_error = (predicted_norm - target_norm).abs()
    norm_relative_error = norm_abs_error / target_norm.clamp_min(1e-8)
    return {
        "goal_mse": float((predicted_goal - target_goal).square().mean()),
        "goal_cosine_similarity": float(
            F.cosine_similarity(predicted_flat, target_flat, dim=1).mean()
        ),
        "goal_patch_cosine_similarity": float(
            F.cosine_similarity(predicted_goal, target_goal, dim=-1).mean()
        ),
        "residual_cosine_similarity": float(
            F.cosine_similarity(residual_pred_flat, residual_target_flat, dim=1).mean()
        ),
        "residual_norm_abs_error": float(norm_abs_error.mean()),
        "residual_norm_relative_error": float(norm_relative_error.mean()),
    }
