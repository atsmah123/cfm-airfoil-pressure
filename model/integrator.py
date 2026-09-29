"""
model/integrator.py

Euler ODE integration for inference.

At inference we integrate:
    dx/dt = v_θ(x_t, t, c)  from t=0 to t=1

Starting from Gaussian noise x_0 ~ N(0,I), we take n_steps Euler steps.
Running this n_samples times gives an ensemble of pressure field samples,
which approximates the posterior p(pressure | sparse_taps, inlet).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


@torch.no_grad()
def euler_integrate(
    vector_field: nn.Module,
    encoder     : nn.Module,
    taps        : torch.Tensor,   # (1, k, 2)
    inlet       : torch.Tensor,   # (1, 3)
    n_surface   : int   = 256,
    n_steps     : int   = 50,
    n_samples   : int   = 100,
    device      : str   = 'cuda',
) -> torch.Tensor:
    """
    Generate n_samples pressure field reconstructions.

    For each sample:
        x_0 ~ N(0, I)          — start from noise
        for each step:
            v = v_θ(x_t, t, c) — predict direction
            x = x + dt * v      — Euler step
        x_1 is one sample from p(pressure | taps, inlet)

    Returns: (n_samples, n_surface) in normalised pressure space
    """
    vector_field.eval()
    encoder.eval()

    # Expand condition for n_samples
    taps_rep  = taps.expand(n_samples, -1, -1).to(device)   # (n_samples, k, 2)
    inlet_rep = inlet.expand(n_samples, -1).to(device)       # (n_samples, 3)

    # Encode condition once — same for all samples
    c = encoder(taps_rep, inlet_rep)                          # (n_samples, 64)

    # Start from noise
    x  = torch.randn(n_samples, n_surface, device=device)

    # Euler integration from t=0 to t=1
    # linspace gives n_steps+1 boundaries; we step from each boundary except the last
    dt     = 1.0 / n_steps
    t_vals = torch.linspace(0, 1 - dt, n_steps, device=device)  # [0, dt, 2dt, ..., 1-dt]

    for t_val in t_vals:
        t_batch = torch.full((n_samples,), t_val, device=device)
        v = vector_field(x, t_batch, c)
        x = x + dt * v

    # Clamp to ±4σ — guards against rare extreme samples
    x = torch.clamp(x, -4.0, 4.0)

    return x  # (n_samples, 256) in normalised space


def compute_cl_cd_from_samples(
    p_denorm : torch.Tensor,   # (n_samples, 256) already in Pa — denormalised
    coords   : torch.Tensor,   # (256, 2)
    normals  : torch.Tensor,   # (256, 2)
    inlet_vel: torch.Tensor,   # (2,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CL and CD for each sample.
    Expects pressure already denormalised in Pa.
    Call as: compute_cl_cd_from_samples(samples * p_std + p_mean, ...)
    """
    p_denorm = p_denorm.numpy()   # (n_samples, 256)

    coords_np    = coords.numpy()
    normals_np   = normals.numpy()
    inlet_np     = inlet_vel.numpy()

    V     = np.linalg.norm(inlet_np)
    q_inf = 0.5 * V**2 + 1e-8

    aoa   = np.arctan2(inlet_np[1], inlet_np[0])
    cos_a, sin_a = np.cos(aoa), np.sin(aoa)

    nx, ny  = normals_np[:, 0], normals_np[:, 1]
    n_lift  = -sin_a * nx + cos_a * ny
    n_drag  =  cos_a * nx + sin_a * ny

    ds = np.linalg.norm(np.diff(coords_np, axis=0), axis=1)
    ds = np.append(ds, np.linalg.norm(coords_np[0] - coords_np[-1]))
    arc = np.concatenate([[0.0], np.cumsum(ds[:-1])])

    # Integrate: ∫ p(s)·n(s) ds along airfoil surface
    CL_samples = np.trapezoid(
        p_denorm * n_lift[None, :], x=arc, axis=1) / q_inf
    CD_samples = np.trapezoid(
        p_denorm * n_drag[None, :], x=arc, axis=1) / q_inf

    return (
        torch.tensor(CL_samples, dtype=torch.float32),
        torch.tensor(CD_samples, dtype=torch.float32),
    )