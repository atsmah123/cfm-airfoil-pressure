"""
data/preprocess.py

Load AirfRANS surface meshes directly from VTP files using pyvista.
Uses VTP line connectivity to order points — robust for all airfoil shapes.

Run:
    python data/preprocess.py --data_root ./data/airfrans_data/Dataset --output_dir ./processed_v3
"""

import argparse
import os
import glob
import random
import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d
from tqdm import tqdm


N_SURFACE   = 256
TRAIN_RATIO = 0.8
SEED        = 42
P_MIN_CLIP  = -5000.0   # sanity check — skip sims with extreme pressures


def load_sim(vtp_path):
    mesh      = pv.read(vtp_path)
    fs_path   = vtp_path.replace('_aerofoil.vtp', '_freestream.vtp')
    fs_mesh   = pv.read(fs_path)
    inlet_vel = fs_mesh.point_data['U'][0, :2]
    return mesh, inlet_vel


def order_by_connectivity(mesh):
    """
    Trace surface points in order using VTP line connectivity.

    The lines array has format: [2, pt_a, pt_b, 2, pt_a, pt_b, ...]
    Each entry means: line segment connecting pt_a to pt_b.
    Every point has exactly 2 neighbors (closed loop).

    We build an adjacency map then walk from the leading edge,
    always moving to the unvisited neighbor.
    """
    lines  = mesh.lines.reshape(-1, 3)[:, 1:]  # (N_cells, 2)
    coords = mesh.points[:, :2]

    # Build adjacency — each point connects to exactly 2 others
    adj = {}
    for a, b in lines:
        a, b = int(a), int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Start at leading edge (min x)
    start   = int(np.argmin(coords[:, 0]))
    order   = [start]
    prev    = -1
    current = start

    for _ in range(mesh.n_points - 1):
        neighbors = [n for n in adj[current] if n != prev]
        if not neighbors:
            break
        nxt     = neighbors[0]
        order.append(nxt)
        prev    = current
        current = nxt

    return np.array(order)


def _normals_from_coords(coords):
    """
    Compute outward-pointing unit normals from ordered surface coordinates.
    Uses central differences: tangent = (next - prev), normal = rotate 90°.
    """
    n = len(coords)
    tangents = coords[(np.arange(n) + 1) % n] - coords[(np.arange(n) - 1) % n]
    # Rotate tangent 90° CCW → inward normal; negate for outward
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def interpolate_to_fixed_grid(coords, pressure, normals, n=N_SURFACE):
    n_pts = len(coords)
    if len(pressure) != n_pts or len(normals) != n_pts:
        raise ValueError(
            f"shape mismatch: coords {n_pts}, pressure {len(pressure)}, normals {len(normals)}"
        )

    diffs = np.diff(coords, axis=0)
    seg   = np.linalg.norm(diffs, axis=1)
    arc   = np.concatenate([[0.0], np.cumsum(seg)])
    arc_n = arc / arc[-1]
    arc_u = np.linspace(0, 1, n)

    p_i = interp1d(arc_n, pressure,       kind='linear')(arc_u)
    c_i = interp1d(arc_n, coords,  axis=0, kind='linear')(arc_u)
    n_i = interp1d(arc_n, normals, axis=0, kind='linear')(arc_u)
    return p_i, c_i, n_i, arc_u


def compute_cl_cd(pressure, coords, normals, inlet_vel):
    V     = np.linalg.norm(inlet_vel)
    q_inf = 0.5 * V**2 + 1e-8
    ds    = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    ds    = np.append(ds, np.linalg.norm(coords[0] - coords[-1]))
    arc   = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    nx, ny   = normals[:, 0], normals[:, 1]
    aoa      = np.arctan2(inlet_vel[1], inlet_vel[0])
    cos_a, sin_a = np.cos(aoa), np.sin(aoa)
    n_lift   = -sin_a * nx + cos_a * ny
    n_drag   =  cos_a * nx + sin_a * ny
    CL = np.trapezoid(pressure * n_lift, x=arc) / q_inf
    CD = np.trapezoid(pressure * n_drag, x=arc) / q_inf
    return float(CL), float(CD)


def preprocess(data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vtp_files = sorted(glob.glob(
        os.path.join(data_root, '**', '*_aerofoil.vtp'), recursive=True
    ))
    print(f"Found {len(vtp_files)} simulations")

    random.seed(SEED)
    random.shuffle(vtp_files)

    n_train = int(len(vtp_files) * TRAIN_RATIO)
    splits  = {'train': vtp_files[:n_train], 'test': vtp_files[n_train:]}

    for split, files in splits.items():
        print(f"\nProcessing {split} ({len(files)} files)...")
        all_pressure, all_coords, all_normals = [], [], []
        all_cl, all_cd, all_inlet = [], [], []
        n_skipped = 0

        for i, vtp_path in enumerate(tqdm(files, desc=split)):
            try:
                mesh, inlet_vel = load_sim(vtp_path)

                # Order by connectivity
                order    = order_by_connectivity(mesh)
                coords   = mesh.points[order, :2]
                pressure = mesh.point_data['p'][order]

                raw_normals = mesh.point_data['Normals']
                if len(raw_normals) == len(order):
                    normals = raw_normals[order, :2]
                else:
                    # Mesh normals array is malformed — recompute from geometry
                    normals = _normals_from_coords(coords)

                p_i, c_i, n_i, _ = interpolate_to_fixed_grid(coords, pressure, normals)

                # Clip extremes after interpolation — handles stalled airfoil spike at LE
                p_i = np.clip(p_i, -5000.0, 5000.0)

                cl, cd = compute_cl_cd(p_i, c_i, n_i, inlet_vel)

                all_pressure.append(p_i)
                all_coords.append(c_i)
                all_normals.append(n_i)
                all_cl.append(cl)
                all_cd.append(cd)
                all_inlet.append(inlet_vel)

            except Exception as e:
                n_skipped += 1
                print(f"  Skipping {os.path.basename(os.path.dirname(vtp_path))}: {e}")

            if (i + 1) % 100 == 0:
                cl_arr = np.array(all_cl)
                cd_arr = np.array(all_cd)
                print(f"  {i+1}/{len(files)} | kept {len(all_cl)} | skipped {n_skipped} | CL: [{cl_arr.min():.3f}, {cl_arr.max():.3f}] | CD: [{cd_arr.min():.4f}, {cd_arr.max():.4f}]")
                np.savez(os.path.join(output_dir, f'{split}_partial.npz'),
                    pressure=np.array(all_pressure, dtype=np.float32),
                    coords  =np.array(all_coords,   dtype=np.float32),
                    normals =np.array(all_normals,  dtype=np.float32),
                    CL      =np.array(all_cl,       dtype=np.float32),
                    CD      =np.array(all_cd,       dtype=np.float32),
                    inlet   =np.array(all_inlet,    dtype=np.float32),
                )

        out = os.path.join(output_dir, f'{split}.npz')
        np.savez(out,
            pressure=np.array(all_pressure, dtype=np.float32),
            coords  =np.array(all_coords,   dtype=np.float32),
            normals =np.array(all_normals,  dtype=np.float32),
            CL      =np.array(all_cl,       dtype=np.float32),
            CD      =np.array(all_cd,       dtype=np.float32),
            inlet   =np.array(all_inlet,    dtype=np.float32),
        )
        print(f"Saved {len(all_pressure)} sims → {out}  (skipped {n_skipped})")
        if len(all_pressure) > 0:
            print(f"  CL range: [{np.array(all_cl).min():.4f}, {np.array(all_cl).max():.4f}]")
            print(f"  CD range: [{np.array(all_cd).min():.4f}, {np.array(all_cd).max():.4f}]")
            p_arr = np.array(all_pressure)
            print(f"  Pressure mean: {p_arr.mean():.1f}, std: {p_arr.std():.1f}")
            print(f"  Pressure min:  {p_arr.min():.1f}, max: {p_arr.max():.1f}")

    train = np.load(os.path.join(output_dir, 'train.npz'))
    np.savez(os.path.join(output_dir, 'stats.npz'),
        p_mean =train['pressure'].mean(),
        p_std  =train['pressure'].std(),
        v_mean =train['inlet'].mean(axis=0),
        v_std  =train['inlet'].std(axis=0),
        cl_mean=train['CL'].mean(),
        cl_std =train['CL'].std(),
        cd_mean=train['CD'].mean(),
        cd_std =train['CD'].std(),
    )
    print(f"\nStats saved → {output_dir}/stats.npz")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./processed_v3')
    args = parser.parse_args()
    preprocess(args.data_root, args.output_dir)