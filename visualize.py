"""
visualize.py

Run:
    python visualize.py --checkpoint ./checkpoints/best.pt --case_idx 5
    python visualize.py --checkpoint ./checkpoints/best.pt --case_idx 5 --random_taps
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.encoder import make_inlet_features
from model.cfm import VectorField
from model.integrator import euler_integrate, compute_cl_cd_from_samples
from dataset import AirfoilDataset
from evaluate import load_models

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS   = {5: '#e74c3c', 10: '#2ecc71', 20: '#3498db', 'truth': '#2c3e50'}
K_VALS   = [5, 10, 20]
AOA_BINS = [(-5, 0), (0, 5), (5, 10), (10, 15)]


def _tap_indices(k, random_taps):
    if random_taps:
        return torch.randperm(256)[:k].sort().values
    return torch.arange(0, 256, 256 // k)[:k]


@torch.no_grad()
def make_plots(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, vf, train_args = load_models(args.checkpoint, device)

    stats  = np.load(os.path.join(train_args.data_dir, 'stats.npz'))
    p_mean = float(stats['p_mean'])
    p_std  = float(stats['p_std'])

    test_ds = AirfoilDataset(
        os.path.join(train_args.data_dir, 'test.npz'),
        os.path.join(train_args.data_dir, 'stats.npz'),
    )
    arc = test_ds.arc_positions.numpy()
    os.makedirs('./results/figures', exist_ok=True)

    # ── Load single case ─────────────────────────────────────────────────────
    idx         = args.case_idx
    p_true_norm = test_ds.pressure_norm[idx].to(device)
    p_true_raw  = test_ds.pressure[idx].numpy()
    coords      = test_ds.coords[idx]
    normals     = test_ds.normals[idx]
    inlet_raw   = test_ds.inlet[idx]
    inlet_norm  = test_ds.inlet_norm[idx]
    CL_true     = test_ds.CL[idx].item()
    CD_true     = test_ds.CD[idx].item()
    inlet_feat  = make_inlet_features(inlet_norm.unsqueeze(0)).to(device)

    # Generate samples for each k — both uniform and random
    samp_uniform = {}
    samp_random  = {}
    ti_uniform   = {}
    ti_random    = {}

    for k in K_VALS:
        for samp_dict, ti_dict, rnd in [
            (samp_uniform, ti_uniform, False),
            (samp_random,  ti_random,  True),
        ]:
            ti = _tap_indices(k, rnd)
            ti_dict[k] = ti
            arc_vals = test_ds.arc_positions[ti].cpu()
            p_vals   = p_true_norm[ti].cpu()
            taps     = torch.stack([arc_vals, p_vals], dim=-1).unsqueeze(0).to(device)
            samples  = euler_integrate(
                vf, encoder, taps, inlet_feat,
                n_surface=256, n_steps=20, n_samples=200, device=str(device),
            )
            samp_dict[k] = samples.cpu().numpy() * p_std + p_mean  # (200, 256) Pa

    # ── Fig 1: Airfoil + numbered taps (top) + pressure reconstruction (bottom)
    # One figure per strategy (uniform / random), 3 columns for k=5,10,20
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe

    coords_np = coords.numpy()  # (256, 2) — x,y surface coords

    for strat_label, samp_dict, ti_dict in [
        ('Uniform', samp_uniform, ti_uniform),
        ('Random',  samp_random,  ti_random),
    ]:
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(f'Airfoil Pressure Reconstruction — {strat_label} Taps  '
                     f'(AoA={np.degrees(np.arctan2(inlet_raw[1].item(), inlet_raw[0].item())):.1f}°, '
                     f'CL={CL_true:.3f}, CD={CD_true:.4f})',
                     fontsize=12)

        # 2 rows × 3 cols, top row taller for airfoil geometry
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.8], hspace=0.45, wspace=0.3)

        for col, k in enumerate(K_VALS):
            ti   = ti_dict[k].numpy()
            samp = samp_dict[k]
            mean = samp.mean(0)
            std  = samp.std(0)

            # ── Top: airfoil geometry with numbered taps ──────────────────
            ax_geo = fig.add_subplot(gs[0, col])
            ax_geo.plot(coords_np[:, 0], coords_np[:, 1],
                        color='#555', lw=1.2, zorder=1)
            ax_geo.fill(coords_np[:, 0], coords_np[:, 1],
                        color='#dde', alpha=0.4, zorder=0)

            tap_xy = coords_np[ti]
            sc = ax_geo.scatter(tap_xy[:, 0], tap_xy[:, 1],
                                c=p_true_raw[ti], cmap='RdBu_r',
                                vmin=p_true_raw.min(), vmax=p_true_raw.max(),
                                s=60, zorder=3, edgecolors='black', lw=0.5)

            # Number each tap
            for n, (tx, ty) in enumerate(tap_xy, start=1):
                ax_geo.text(tx, ty + 0.008, str(n), ha='center', va='bottom',
                            fontsize=6, fontweight='bold', color='black',
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                        foreground='white')])

            ax_geo.set_aspect('equal')
            ax_geo.set_title(f'k = {k} taps', fontsize=11)
            ax_geo.set_xlabel('x/c', fontsize=9)
            if col == 0:
                ax_geo.set_ylabel('y/c', fontsize=9)
            ax_geo.tick_params(labelsize=8)

            # ── Bottom: pressure reconstruction ───────────────────────────
            ax_p = fig.add_subplot(gs[1, col])
            ax_p.fill_between(arc, mean - 2*std, mean + 2*std,
                              alpha=0.2, color=COLORS[k], label='±2σ')
            ax_p.fill_between(arc, mean - std,   mean + std,
                              alpha=0.4, color=COLORS[k], label='±1σ')
            ax_p.plot(arc, mean,       color=COLORS[k],       lw=2,
                      label='Mean pred.')
            ax_p.plot(arc, p_true_raw, color=COLORS['truth'], lw=1.5,
                      ls='--', label='Ground truth')

            # Tap markers with numbers matching the airfoil plot
            for n, t in enumerate(ti, start=1):
                ax_p.scatter(arc[t], p_true_raw[t],
                             color='black', zorder=5, s=35)
                ax_p.text(arc[t], p_true_raw[t] + (p_true_raw.max()-p_true_raw.min())*0.04,
                          str(n), ha='center', va='bottom', fontsize=6,
                          fontweight='bold',
                          path_effects=[pe.withStroke(linewidth=1.5,
                                                      foreground='white')])

            ax_p.set_xlabel('Arc-length s', fontsize=9)
            if col == 0:
                ax_p.set_ylabel('Pressure (Pa)', fontsize=9)
            ax_p.tick_params(labelsize=8)

        # Shared legend at bottom
        legend_els = [
            Line2D([0], [0], color=COLORS['truth'], ls='--', lw=1.5, label='Ground truth'),
            Line2D([0], [0], color='grey', lw=2, label='Mean pred.'),
            plt.Rectangle((0,0),1,1, fc='grey', alpha=0.4, label='±1σ'),
            plt.Rectangle((0,0),1,1, fc='grey', alpha=0.2, label='±2σ'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=7, label='Tap (numbered)'),
        ]
        fig.legend(handles=legend_els, loc='lower center', ncol=5,
                   bbox_to_anchor=(0.5, -0.01), fontsize=9)

        fname = f'./results/figures/fig1_{strat_label.lower()}_reconstruction.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved → {fname}")

    # ── Fig 2: Relative error across k (best strategy from args) ─────────────
    samp_dict = samp_random if args.random_taps else samp_uniform
    ti_dict   = ti_random   if args.random_taps else ti_uniform
    label     = 'Random' if args.random_taps else 'Uniform'

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(f'Relative Reconstruction Error — {label} Taps (case {idx})', fontsize=13)

    for ax, k in zip(axes, K_VALS):
        samp = samp_dict[k]
        mean = samp.mean(0)
        std  = samp.std(0)
        p_range = p_true_raw.max() - p_true_raw.min()
        rel_err = np.abs(mean - p_true_raw) / (p_range + 1e-6) * 100  # %

        ax.fill_between(arc,
                        (mean - 2*std - p_true_raw) / (p_range + 1e-6) * 100,
                        (mean + 2*std - p_true_raw) / (p_range + 1e-6) * 100,
                        alpha=0.2, color=COLORS[k], label='±2σ band')
        ax.plot(arc, rel_err, color=COLORS[k], lw=2, label='|Mean error| %')
        ax.axhline(0, color='black', lw=0.8, ls='--')

        ti = ti_dict[k].numpy()
        ax.scatter(arc[ti], rel_err[ti], color='black', zorder=5, s=30, label='Taps')

        mae_pct = rel_err.mean()
        ax.set_title(f'k={k}  (MAE={mae_pct:.1f}% of range)')
        ax.set_xlabel('Arc-length position')
        if ax is axes[0]:
            ax.set_ylabel('Relative error (% of pressure range)')

    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('./results/figures/fig2_relative_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → ./results/figures/fig2_relative_error.png")

    # ── Fig 3: CL/CD posteriors ───────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'CL and CD Posteriors vs Tap Count — {label} Taps')

    cl_data, cd_data = [], []
    for k in K_VALS:
        samp_pa    = torch.from_numpy(samp_dict[k])
        CL_s, CD_s = compute_cl_cd_from_samples(samp_pa, coords, normals, inlet_raw)
        cl_data.append(CL_s.numpy())
        cd_data.append(CD_s.numpy())

    vp1 = ax1.violinplot(cl_data, positions=K_VALS, widths=3, showmedians=True)
    vp2 = ax2.violinplot(cd_data, positions=K_VALS, widths=3, showmedians=True)
    for vp in [vp1, vp2]:
        for pc, k in zip(vp['bodies'], K_VALS):
            pc.set_facecolor(COLORS[k]); pc.set_alpha(0.6)

    ax1.axhline(CL_true, color=COLORS['truth'], ls='--', lw=1.5, label='Ground truth')
    ax2.axhline(CD_true, color=COLORS['truth'], ls='--', lw=1.5, label='Ground truth')
    ax1.set_xlabel('k'); ax1.set_ylabel('CL'); ax1.set_title('Lift coefficient'); ax1.legend()
    ax2.set_xlabel('k'); ax2.set_ylabel('CD'); ax2.set_title('Drag coefficient'); ax2.legend()
    plt.tight_layout()
    plt.savefig('./results/figures/fig3_cl_cd_posteriors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → ./results/figures/fig3_cl_cd_posteriors.png")

    # ── Fig 4: MAE by AoA — best k across all test cases ─────────────────────
    best_k = 10   # use k=10 as representative best
    print(f"\nComputing per-case MAE by AoA (k={best_k}, {label} taps, all {len(test_ds)} test cases)...")

    aoa_list, mae_list, rel_mae_list = [], [], []
    for i in range(len(test_ds)):
        p_n      = test_ds.pressure_norm[i].to(device)
        p_raw    = test_ds.pressure[i].numpy()
        il       = make_inlet_features(test_ds.inlet_norm[i].unsqueeze(0)).to(device)
        inlet_r  = test_ds.inlet[i]
        aoa      = float(np.degrees(np.arctan2(inlet_r[1].item(), inlet_r[0].item())))

        ti       = _tap_indices(best_k, args.random_taps)
        av       = test_ds.arc_positions[ti].cpu()
        pv       = p_n[ti].cpu()
        taps_    = torch.stack([av, pv], dim=-1).unsqueeze(0).to(device)

        samp     = euler_integrate(vf, encoder, taps_, il,
                                   n_surface=256, n_steps=20,
                                   n_samples=50, device=str(device))
        p_pred   = samp.cpu().mean(0).numpy() * p_std + p_mean
        mae      = float(np.abs(p_pred - p_raw).mean())
        p_range  = float(p_raw.max() - p_raw.min()) + 1e-6
        aoa_list.append(aoa)
        mae_list.append(mae)
        rel_mae_list.append(mae / p_range * 100)

    aoa_arr     = np.array(aoa_list)
    mae_arr     = np.array(mae_list)
    rel_mae_arr = np.array(rel_mae_list)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Reconstruction MAE vs Angle of Attack — k={best_k}, {label} Taps', fontsize=13)

    # Scatter + bin means
    for ax, vals, ylabel in [
        (axes[0], mae_arr,     'MAE (Pa)'),
        (axes[1], rel_mae_arr, 'Relative MAE (% of pressure range)'),
    ]:
        ax.scatter(aoa_arr, vals, alpha=0.5, s=20, color='steelblue', label='Test cases')

        # Bin means
        bin_edges  = np.array([-5, 0, 5, 10, 15], dtype=float)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means   = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            sub = vals[(aoa_arr >= lo) & (aoa_arr < hi)]
            bin_means.append(sub.mean() if len(sub) else np.nan)
        ax.plot(bin_centers, bin_means, 'o-', color='#e74c3c',
                lw=2, ms=8, label='Bin mean')

        ax.set_xlabel('Angle of Attack (°)')
        ax.set_ylabel(ylabel)
        ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('./results/figures/fig4_mae_by_aoa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → ./results/figures/fig4_mae_by_aoa.png")

    # ── Fig 5: MAE distribution across test set (all k) ──────────────────────
    print(f"\nComputing full test-set MAE distribution for all k...")

    all_maes = {k: [] for k in K_VALS}
    for i in range(len(test_ds)):
        p_n   = test_ds.pressure_norm[i].to(device)
        p_raw = test_ds.pressure[i].numpy()
        il    = make_inlet_features(test_ds.inlet_norm[i].unsqueeze(0)).to(device)

        for k in K_VALS:
            ti    = _tap_indices(k, args.random_taps)
            av    = test_ds.arc_positions[ti].cpu()
            pv    = p_n[ti].cpu()
            taps_ = torch.stack([av, pv], dim=-1).unsqueeze(0).to(device)
            samp  = euler_integrate(vf, encoder, taps_, il,
                                    n_surface=256, n_steps=20,
                                    n_samples=50, device=str(device))
            p_pred = samp.cpu().mean(0).numpy() * p_std + p_mean
            all_maes[k].append(float(np.abs(p_pred - p_raw).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'MAE Distribution Across Test Set — {label} Taps', fontsize=13)

    # Box + strip plot
    ax = axes[0]
    data = [all_maes[k] for k in K_VALS]
    bp   = ax.boxplot(data, positions=K_VALS, widths=2.5,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color='black', lw=2))
    for patch, k in zip(bp['boxes'], K_VALS):
        patch.set_facecolor(COLORS[k]); patch.set_alpha(0.6)
    # Overlay individual points
    for k, vals in all_maes.items():
        jitter = np.random.uniform(-0.8, 0.8, len(vals))
        ax.scatter(np.full(len(vals), k) + jitter, vals,
                   alpha=0.3, s=8, color=COLORS[k])
    ax.set_xlabel('Number of taps k')
    ax.set_ylabel('MAE (Pa)')
    ax.set_title('Absolute MAE distribution')
    ax.set_xticks(K_VALS)

    # CDF plot
    ax = axes[1]
    for k in K_VALS:
        vals_sorted = np.sort(all_maes[k])
        cdf = np.arange(1, len(vals_sorted)+1) / len(vals_sorted)
        ax.plot(vals_sorted, cdf, color=COLORS[k], lw=2, label=f'k={k}')
        # Mark median and p90
        for pct, ls in [(50, '--'), (90, ':')]:
            pv = np.percentile(vals_sorted, pct)
            ax.axvline(pv, color=COLORS[k], lw=0.8, ls=ls, alpha=0.6)
    ax.set_xlabel('MAE (Pa)')
    ax.set_ylabel('Cumulative fraction of test cases')
    ax.set_title('CDF of MAE (dashed=median, dotted=p90)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('./results/figures/fig5_mae_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → ./results/figures/fig5_mae_distribution.png")

    print("\nAll figures saved → ./results/figures/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--case_idx',    type=int, default=5)
    parser.add_argument('--random_taps', action='store_true', default=False)
    args = parser.parse_args()
    make_plots(args)
