import torch
import numpy as np
from train import AirfoilDataset
from model.encoder import SparseEncoder, make_inlet_features
from model.cfm import VectorField
from model.integrator import euler_integrate

device = torch.device('cuda')
ckpt   = torch.load('./checkpoints_v5/best.pt', weights_only=False, map_location=device)

encoder = SparseEncoder(tap_input_dim=2, inlet_dim=3, hidden_dim=64, cond_dim=64).to(device)
vf      = VectorField(n_surface=256, cond_dim=64, time_dim=16, hidden_dim=256, n_layers=3).to(device)
encoder.load_state_dict(ckpt['encoder'])
vf.load_state_dict(ckpt['vf'])

stats   = np.load('./processed_v4/stats.npz')
p_mean  = float(stats['p_mean'])
p_std   = float(stats['p_std'])
ds      = AirfoilDataset('./processed_v4/test.npz', './processed_v4/stats.npz')

idx          = 0
p_true_norm  = ds.pressure_norm[idx].to(device)
inlet_feat   = make_inlet_features(ds.inlet_norm[idx].unsqueeze(0)).to(device)

# k=20 taps
tap_indices = torch.arange(0, 256, 256//20)[:20]
arc_vals    = ds.arc_positions[tap_indices].cpu()
p_vals      = p_true_norm[tap_indices].cpu()
taps        = torch.stack([arc_vals, p_vals], dim=-1).unsqueeze(0).to(device)

samples = euler_integrate(vf, encoder, taps, inlet_feat,
                          n_surface=256, n_steps=20, n_samples=50, device='cuda')

p_denorm = samples.cpu() * p_std + p_mean
print(f"Samples range (Pa): [{p_denorm.min():.1f}, {p_denorm.max():.1f}]")
print(f"Expected:           [-5000, 4375] Pa")
print(f"Ground truth range: [{(ds.pressure[idx]).min():.1f}, {(ds.pressure[idx]).max():.1f}]")