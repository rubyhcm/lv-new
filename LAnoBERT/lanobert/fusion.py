"""Gated fusion of frozen LAnoBERT and Hybrid-RAG representations."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Fuse baseline, semantic-context, and hard-state representations."""

    def __init__(self, baseline_dim: int, context_dim: int,
                 hard_state_dim: int, hidden_dim: int = 64):
        super().__init__()
        if min(baseline_dim, context_dim, hard_state_dim, hidden_dim) <= 0:
            raise ValueError("all representation dimensions must be positive")
        self.baseline_projection = nn.Linear(baseline_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.hard_state_projection = nn.Linear(hard_state_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.output = nn.Linear(hidden_dim, 1)
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, baseline: torch.Tensor, context: torch.Tensor,
                hard_state: torch.Tensor) -> torch.Tensor:
        baseline_h = torch.tanh(self.baseline_projection(baseline))
        context_h = torch.tanh(self.context_projection(context))
        hard_state_h = torch.tanh(self.hard_state_projection(hard_state))
        stacked = torch.stack((baseline_h, context_h, hard_state_h), dim=1)
        weights = torch.softmax(self.gate(torch.cat(
            (baseline_h, context_h, hard_state_h), dim=-1
        )), dim=-1)
        self.last_gate = weights.detach()
        return self.output((stacked * weights.unsqueeze(-1)).sum(dim=1)).squeeze(-1)


def freeze_baseline(model: nn.Module) -> nn.Module:
    """Freeze a LAnoBERT model before training only the fusion layer."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model
