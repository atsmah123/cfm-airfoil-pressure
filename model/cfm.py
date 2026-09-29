"""
model/cfm.py

Conv UNet vector field for Conditional Flow Matching.

Architecture:
    The pressure field (B, 256) is treated as a 1D spatial signal (B, 1, 256).
    A 3-level UNet encodes it to coarser resolutions and decodes back, with
    skip connections preserving fine-scale detail.

    Every residual conv block is FiLM-conditioned on (c, t_emb), so the
    condition is injected at every spatial scale — not just the bottleneck.

    Circular padding throughout: arc=0 and arc=1 are the same physical point.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    assert dim % 2 == 0
    half  = dim // 2
    freqs = 2 ** torch.arange(half, device=t.device).float()
    args  = t.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """
    1D residual conv block with FiLM conditioning.
    h_out = h + act(GroupNorm(conv(h)) * (1 + scale) + shift)
    """

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv  = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, padding_mode='circular')
        self.norm  = nn.GroupNorm(8, channels)
        self.film  = nn.Linear(cond_dim, channels * 2)
        self.act   = nn.SiLU()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) → scale/shift: (B, C, 1) for broadcast over length
        scale, shift = self.film(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        h2 = self.norm(self.conv(h))
        h2 = h2 * (1 + scale) + shift
        return h + self.act(h2)


class VectorField(nn.Module):
    """
    3-level conv UNet.

    Resolution levels: 256 → 128 → 64 (bottleneck)
    Each level has n_blocks ResBlocks, all FiLM-conditioned.
    Skip connections from encoder to decoder at each level.
    """

    def __init__(
        self,
        cond_dim : int = 128,
        time_dim : int = 32,
        base_ch  : int = 64,   # channels at finest level; doubles each level
        n_blocks : int = 2,    # ResBlocks per level
        n_surface: int = 256,  # kept for API consistency, not used internally
    ):
        super().__init__()
        self.time_dim = time_dim
        film_dim = cond_dim + time_dim

        ch = [base_ch, base_ch * 2, base_ch * 4]  # [64, 128, 256]

        # Input/output projections (operate on channels, not spatial)
        self.in_proj  = nn.Conv1d(1, ch[0], kernel_size=1)
        self.out_proj = nn.Conv1d(ch[0], 1, kernel_size=1)

        # Encoder
        self.enc0 = nn.ModuleList([ResBlock(ch[0], film_dim) for _ in range(n_blocks)])
        self.down0 = nn.Conv1d(ch[0], ch[1], kernel_size=4, stride=2, padding=1)  # 256→128

        self.enc1 = nn.ModuleList([ResBlock(ch[1], film_dim) for _ in range(n_blocks)])
        self.down1 = nn.Conv1d(ch[1], ch[2], kernel_size=4, stride=2, padding=1)  # 128→64

        # Bottleneck
        self.mid = nn.ModuleList([ResBlock(ch[2], film_dim) for _ in range(n_blocks)])

        # Decoder — skip projection first, then ResBlocks at reduced channels
        self.up1   = nn.ConvTranspose1d(ch[2], ch[1], kernel_size=4, stride=2, padding=1)
        self.skip1 = nn.Conv1d(ch[1] * 2, ch[1], kernel_size=1)
        self.dec1  = nn.ModuleList([ResBlock(ch[1], film_dim) for _ in range(n_blocks)])

        self.up0   = nn.ConvTranspose1d(ch[1], ch[0], kernel_size=4, stride=2, padding=1)
        self.skip0 = nn.Conv1d(ch[0] * 2, ch[0], kernel_size=1)
        self.dec0  = nn.ModuleList([ResBlock(ch[0], film_dim) for _ in range(n_blocks)])

    def _run_blocks(self, h, blocks, cond):
        for blk in blocks:
            h = blk(h, cond)
        return h

    def forward(
        self,
        x_t: torch.Tensor,  # (B, 256)
        t  : torch.Tensor,  # (B,)
        c  : torch.Tensor,  # (B, cond_dim)
    ) -> torch.Tensor:

        t_emb = time_embedding(t, self.time_dim)
        cond  = torch.cat([c, t_emb], dim=-1)          # (B, film_dim)

        h = self.in_proj(x_t.unsqueeze(1))             # (B, ch0, 256)

        # Encode
        h0 = self._run_blocks(h,  self.enc0, cond)     # (B, ch0, 256)
        h1 = self._run_blocks(self.down0(h0), self.enc1, cond)  # (B, ch1, 128)
        h2 = self._run_blocks(self.down1(h1), self.mid,  cond)  # (B, ch2,  64)

        # Decode: upsample → concat skip → project → ResBlocks
        h = self.skip1(torch.cat([self.up1(h2), h1], dim=1))  # (B, ch1, 128)
        h = self._run_blocks(h, self.dec1, cond)

        h = self.skip0(torch.cat([self.up0(h), h0], dim=1))   # (B, ch0, 256)
        h = self._run_blocks(h, self.dec0, cond)

        return self.out_proj(h).squeeze(1)             # (B, 256)


def cfm_loss(
    vector_field: VectorField,
    encoder     : nn.Module,
    x_1         : torch.Tensor,
    taps        : torch.Tensor,
    inlet       : torch.Tensor,
    k           : torch.Tensor | None = None,
    sigma_min   : float = 1e-4,
) -> torch.Tensor:
    """
    Straight-line OT path CFM loss.
    x_t = (1 - (1-sigma_min)*t) * x_0 + t * x_1
    u_t = x_1 - (1-sigma_min) * x_0
    """
    B, device = x_1.shape[0], x_1.device

    x_0      = torch.randn_like(x_1)
    t        = torch.rand(B, device=device)
    t_expand = t.view(B, 1)

    x_t = (1 - (1 - sigma_min) * t_expand) * x_0 + t_expand * x_1
    u_t = x_1 - (1 - sigma_min) * x_0

    c      = encoder(taps, inlet, k)
    v_pred = vector_field(x_t, t, c)

    return F.mse_loss(v_pred, u_t)
