"""
evaluate.py

Run:
    python evaluate.py --checkpoint ./checkpoints/best.pt --k 5 10 20
    python evaluate.py --checkpoint ./checkpoints/best.pt --k 5 10 20 --random_taps
"""

import argparse
import os
import numpy as np
import torch

from model.encoder import SparseEncoder, make_inlet_features
from model.cfm import VectorField
from model.integrator import euler_integrate, compute_cl_cd_from_samples
from dataset import AirfoilDataset


def load_models(checkpoint_path, device):
    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args    = argparse.Namespace(**ckpt['args'])

    encoder = SparseEncoder(
        tap_input_dim=2, inlet_dim=3,
        hidden_dim=128, cond_dim=128,
    ).to(device)

    vf = VectorField(
        n_surface=256, cond_dim=128,
        time_dim=32, base_ch=64, n_blocks=2,
    ).to(device)

    encoder.load_state_dict(ckpt['encoder'])
    vf.load_state_dict(ckpt['vf'])
    encoder.eval(); vf.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.4f})")
    return encoder, vf, args


def _tap_indices(k, random_taps):
    if random_taps:
        return torch.randperm(256)[:k].sort().values
    return torch.arange(0, 256, 256 // k)[:k]


@torch.no_grad()
def evaluate(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, vf, train_args = load_models(args.checkpoint, device)

    stats  = np.load(os.path.join(train_args.data_dir, 'stats.npz'))
    p_mean = float(stats['p_mean'])
    p_std  = float(stats['p_std'])

    test_ds = AirfoilDataset(
        os.path.join(train_args.data_dir, 'test.npz'),
        os.path.join(train_args.data_dir, 'stats.npz'),
    )

    n_eval = min(args.n_eval, len(test_ds))
    print(f"Evaluating {n_eval} test cases with {args.n_samples} samples each...")

    # Per-case records keyed by k
    records = {k: [] for k in args.k}

    for i in range(n_eval):
        p_true_norm = test_ds.pressure_norm[i].to(device)
        p_true_raw  = test_ds.pressure[i].numpy()
        coords      = test_ds.coords[i]
        normals     = test_ds.normals[i]
        inlet_raw   = test_ds.inlet[i]
        inlet_norm  = test_ds.inlet_norm[i]
        CL_true     = test_ds.CL[i].item()
        CD_true     = test_ds.CD[i].item()
        aoa         = float(np.degrees(np.arctan2(inlet_raw[1].item(), inlet_raw[0].item())))

        inlet_feat = make_inlet_features(inlet_norm.unsqueeze(0)).to(device)

        for k in args.k:
            ti       = _tap_indices(k, args.random_taps)
            arc_vals = test_ds.arc_positions[ti].cpu()
            p_vals   = p_true_norm[ti].cpu()
            taps     = torch.stack([arc_vals, p_vals], dim=-1).unsqueeze(0).to(device)

            samples  = euler_integrate(
                vf, encoder, taps, inlet_feat,
                n_surface=256, n_steps=args.n_steps,
                n_samples=args.n_samples, device=str(device),
            )
            p_samp = samples.cpu() * p_std + p_mean   # (n_samples, 256) Pa
            p_mean_pred = p_samp.mean(0).numpy()

            mae      = float(np.abs(p_mean_pred - p_true_raw).mean())
            # Relative MAE: normalised by per-case pressure range (avoids division by ~0)
            p_range  = float(p_true_raw.max() - p_true_raw.min()) + 1e-6
            rel_mae  = mae / p_range

            CL_s, CD_s = compute_cl_cd_from_samples(
                torch.from_numpy(p_samp.numpy()), coords, normals, inlet_raw)
            cl_mean = CL_s.mean().item()
            cd_mean = CD_s.mean().item()
            cl_std  = CL_s.std().item()
            cd_std  = CD_s.std().item()

            records[k].append({
                'case'       : i,
                'aoa'        : aoa,
                'CL_true'    : CL_true,
                'CD_true'    : CD_true,
                'mae'        : mae,
                'rel_mae'    : rel_mae,
                'cl_err'     : abs(cl_mean - CL_true),
                'cd_err'     : abs(cd_mean - CD_true),
                'cl_std'     : cl_std,
                'cd_std'     : cd_std,
                'cl_covered' : int(abs(cl_mean - CL_true) < 2 * cl_std),
                'cd_covered' : int(abs(cd_mean - CD_true) < 2 * cd_std),
            })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_eval} done")

    # ── Overall summary ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'k':>4} | {'MAE(p)':>10} | {'RelMAE':>8} | {'MAE(CL)':>8} | {'MAE(CD)':>8} | {'CL cov%':>8} | {'CD cov%':>8}")
    print("-"*80)
    for k in args.k:
        r = records[k]
        maes     = [x['mae']        for x in r]
        rel_maes = [x['rel_mae']    for x in r]
        cl_errs  = [x['cl_err']     for x in r]
        cd_errs  = [x['cd_err']     for x in r]
        cl_covs  = [x['cl_covered'] for x in r]
        cd_covs  = [x['cd_covered'] for x in r]
        print(
            f"{k:>4} | "
            f"{np.mean(maes):>10.4f} | "
            f"{np.mean(rel_maes):>7.1%} | "
            f"{np.mean(cl_errs):>8.4f} | "
            f"{np.mean(cd_errs):>8.4f} | "
            f"{100*np.mean(cl_covs):>7.1f}% | "
            f"{100*np.mean(cd_covs):>7.1f}%"
        )
    print("="*80)
    print("Coverage target: ~95% for well-calibrated ±2σ intervals")

    # ── AoA breakdown for best k ──────────────────────────────────────────────
    best_k = args.k[int(np.argmin([np.mean([x['mae'] for x in records[k]]) for k in args.k]))]
    print(f"\n── MAE(p) by AoA bin  (best k={best_k}) ──────────────────────────────────")
    aoa_bins = [(-5, 0), (0, 5), (5, 10), (10, 15)]
    print(f"{'AoA range':>15} | {'n':>4} | {'Mean MAE':>10} | {'Median MAE':>10} | {'p90 MAE':>10} | {'RelMAE':>8}")
    print("-"*70)
    for lo, hi in aoa_bins:
        sub = [x for x in records[best_k] if lo <= x['aoa'] < hi]
        if not sub:
            continue
        maes = [x['mae'] for x in sub]
        rels = [x['rel_mae'] for x in sub]
        print(
            f"  [{lo:+d}°, {hi:+d}°)    | "
            f"{len(sub):>4} | "
            f"{np.mean(maes):>10.1f} | "
            f"{np.median(maes):>10.1f} | "
            f"{np.percentile(maes,90):>10.1f} | "
            f"{np.mean(rels):>7.1%}"
        )

    # ── Distribution stats across test set ───────────────────────────────────
    print(f"\n── MAE(p) distribution across test set ───────────────────────────────────")
    print(f"{'k':>4} | {'p10':>8} | {'p25':>8} | {'Median':>8} | {'p75':>8} | {'p90':>8} | {'p95':>8} | {'Max':>8}")
    print("-"*70)
    for k in args.k:
        maes = sorted([x['mae'] for x in records[k]])
        print(
            f"{k:>4} | "
            f"{np.percentile(maes,10):>8.1f} | "
            f"{np.percentile(maes,25):>8.1f} | "
            f"{np.percentile(maes,50):>8.1f} | "
            f"{np.percentile(maes,75):>8.1f} | "
            f"{np.percentile(maes,90):>8.1f} | "
            f"{np.percentile(maes,95):>8.1f} | "
            f"{np.max(maes):>8.1f}"
        )

    os.makedirs('./results', exist_ok=True)
    np.save('./results/eval_records.npy', records, allow_pickle=True)
    print("\nResults saved → ./results/eval_records.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--k',           type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--n_samples',   type=int, default=100)
    parser.add_argument('--n_steps',     type=int, default=20)
    parser.add_argument('--n_eval',      type=int, default=100)
    parser.add_argument('--random_taps', action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args)
