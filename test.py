import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Load ──────────────────────────────────────────────────────────────────────
path = './data/airfrans_data/Dataset/airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616/airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616_aerofoil.vtp'
fs_path = path.replace('_aerofoil.vtp', '_freestream.vtp')

mesh     = pv.read(path)
fs_mesh  = pv.read(fs_path)

coords   = mesh.points[:, :2]
pressure = mesh.point_data['p']
normals  = mesh.point_data['Normals'][:, :2]
inlet_vel = fs_mesh.point_data['U'][0, :2]

print(f"Inlet velocity: {inlet_vel}")
print(f"AoA (deg): {np.degrees(np.arctan2(inlet_vel[1], inlet_vel[0])):.3f}")
print(f"N points: {len(coords)}")

# ── Angular ordering ──────────────────────────────────────────────────────────
cx, cy  = coords[:, 0].mean(), coords[:, 1].mean()
angles  = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
order   = np.argsort(angles)

coords   = coords[order]
pressure = pressure[order]
normals  = normals[order]

# Roll to leading edge
le_idx   = np.argmin(coords[:, 0])
coords   = np.roll(coords,   -le_idx, axis=0)
pressure = np.roll(pressure, -le_idx)
normals  = np.roll(normals,  -le_idx, axis=0)

# ── Check ordering visually ───────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap='viridis', s=10)
plt.colorbar(label='point index after ordering')
plt.title('Point ordering after angular sort (should go smoothly around airfoil)')
plt.axis('equal')
plt.savefig('ordering_after.png', dpi=100)
plt.close()
print("Saved ordering_after.png")

# ── Interpolate to 256 points ─────────────────────────────────────────────────
N = 256
diffs = np.diff(coords, axis=0)
seg   = np.linalg.norm(diffs, axis=1)
arc   = np.concatenate([[0.0], np.cumsum(seg)])
arc_n = arc / arc[-1]
arc_u = np.linspace(0, 1, N)

p_i = interp1d(arc_n, pressure,       kind='linear')(arc_u)
c_i = interp1d(arc_n, coords,  axis=0, kind='linear')(arc_u)
n_i = interp1d(arc_n, normals, axis=0, kind='linear')(arc_u)

print(f"\nInterpolated pressure range: [{p_i.min():.2f}, {p_i.max():.2f}]")

# ── CL / CD ───────────────────────────────────────────────────────────────────
V     = np.linalg.norm(inlet_vel)
q_inf = 0.5 * V**2 + 1e-8
ds    = np.linalg.norm(np.diff(c_i, axis=0), axis=1)
ds    = np.append(ds, np.linalg.norm(c_i[0] - c_i[-1]))

nx, ny   = n_i[:, 0], n_i[:, 1]
aoa      = np.arctan2(inlet_vel[1], inlet_vel[0])
cos_a, sin_a = np.cos(aoa), np.sin(aoa)
n_lift   = -sin_a * nx + cos_a * ny
n_drag   =  cos_a * nx + sin_a * ny

CL = np.trapezoid(p_i * n_lift * ds) / q_inf
CD = np.trapezoid(p_i * n_drag * ds) / q_inf

print(f"CL: {CL:.4f}  (expected ~0.065)")
print(f"CD: {CD:.4f}  (expected ~0.0016)")

# ── Plot pressure field ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(arc_u, p_i)
plt.xlabel('Arc-length position')
plt.ylabel('Pressure (Pa)')
plt.title('Surface pressure distribution after interpolation')
plt.savefig('pressure_check.png', dpi=100)
plt.close()
print("Saved pressure_check.png")