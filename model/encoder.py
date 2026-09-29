"""
model/encoder.py

DeepSets encoder for sparse pressure taps.

Taps are at arbitrary arc positions (randomly sampled during training), so the
encoder must learn which positions are informative. Each tap (arc_position,
normalised_pressure) is processed by a shared MLP, then aggregated via softmax
attention weighted by arc position. This allows the model to upweight taps near
high-information regions (suction peak, trailing edge) regardless of where they fall.

The pooled embedding is concatenated with inlet features [vx, vy, AoA] and
k/k_max (normalised tap count, so the vector field can learn to sharpen the
posterior when more observations are available).
"""

import torch
import torch.nn as nn


class SparseEncoder(nn.Module):

    def __init__(
        self,
        tap_input_dim: int = 2,
        inlet_dim    : int = 3,
        hidden_dim   : int = 128,
        cond_dim     : int = 128,
    ):
        super().__init__()

        self.tap_mlp = nn.Sequential(
            nn.Linear(tap_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Learned scalar attention weight from arc position
        self.attn = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # +1 for normalised tap count k/k_max
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + inlet_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(
        self,
        taps : torch.Tensor,                # (B, max_k, 2)
        inlet: torch.Tensor,                # (B, 3)
        k    : torch.Tensor | None = None,  # (B,) actual tap counts
    ) -> torch.Tensor:

        h   = self.tap_mlp(taps)            # (B, max_k, hidden_dim)
        arc = taps[:, :, 0:1]              # (B, max_k, 1)
        w   = self.attn(arc)               # (B, max_k, 1)

        if k is not None:
            max_k = taps.shape[1]
            idx   = torch.arange(max_k, device=taps.device).unsqueeze(0)
            mask  = (idx < k.unsqueeze(1)).unsqueeze(-1).float()
            w     = w.masked_fill(mask == 0, float('-inf'))

        w = torch.softmax(w, dim=1)        # (B, max_k, 1)
        h = (h * w).sum(dim=1)             # (B, hidden_dim)

        if k is not None:
            k_norm = k.float().unsqueeze(-1) / 20.0
        else:
            k_norm = torch.full((taps.shape[0], 1),
                                taps.shape[1] / 20.0, device=taps.device)

        h = torch.cat([h, inlet, k_norm], dim=-1)
        return self.fusion(h)              # (B, cond_dim)


def make_inlet_features(inlet_vel: torch.Tensor) -> torch.Tensor:
    """[vx, vy] → [vx, vy, AoA]"""
    aoa = torch.atan2(inlet_vel[:, 1], inlet_vel[:, 0]).unsqueeze(-1)
    return torch.cat([inlet_vel, aoa], dim=-1)
