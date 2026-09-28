# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Shared MLP decoder for MiniCPM-predicted DINOv2 patch residuals."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualMLPBlock(nn.Module):
    """Width-preserving MLP block with a residual connection."""

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.linear(self.norm(x)))


class DINOResidualHead(nn.Module):
    """Decode each VLM query state into a signed DINO patch-feature residual.

    The same decoder is shared across all 256 patch positions. It intentionally
    accepts only VLM hidden states; current DINO features are added downstream
    to form the predicted goal and never enter this module.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 384,
        num_residual_blocks: int = 2,
        num_tokens: int = 256,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, num_tokens) <= 0:
            raise ValueError("DINOResidualHead dimensions and token count must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.residual_blocks = nn.ModuleList(
            _ResidualMLPBlock(self.hidden_dim) for _ in range(num_residual_blocks)
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, query_hidden_states: torch.Tensor) -> torch.Tensor:
        if query_hidden_states.ndim != 3:
            raise ValueError(
                "query_hidden_states must have shape [batch, patches, hidden], "
                f"got {tuple(query_hidden_states.shape)}"
            )
        batch, patches, hidden = query_hidden_states.shape
        if patches != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} residual tokens, got {patches}")
        if hidden != self.input_dim:
            raise ValueError(f"expected VLM width {self.input_dim}, got {hidden}")

        x = self.activation(self.input_projection(self.input_norm(query_hidden_states)))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_projection(x).reshape(batch, self.num_tokens, self.output_dim)
