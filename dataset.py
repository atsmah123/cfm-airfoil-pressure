"""
dataset.py

AirfoilDataset and collate_fn — separated from train.py to avoid circular imports.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from model.encoder import make_inlet_features


class AirfoilDataset(Dataset):

    def __init__(self, npz_path, stats_path, k_min=5, k_max=20, random_taps=True):
        data  = np.load(npz_path)
        stats = np.load(stats_path)

        self.pressure = torch.tensor(data['pressure'], dtype=torch.float32)
        self.coords   = torch.tensor(data['coords'],   dtype=torch.float32)
        self.normals  = torch.tensor(data['normals'],  dtype=torch.float32)
        self.inlet    = torch.tensor(data['inlet'],    dtype=torch.float32)
        self.CL       = torch.tensor(data['CL'],       dtype=torch.float32)
        self.CD       = torch.tensor(data['CD'],       dtype=torch.float32)

        self.p_mean = float(stats['p_mean'])
        self.p_std  = float(stats['p_std'])
        self.v_mean = torch.tensor(stats['v_mean'], dtype=torch.float32)
        self.v_std  = torch.tensor(stats['v_std'],  dtype=torch.float32)

        self.pressure_norm = (self.pressure - self.p_mean) / (self.p_std + 1e-8)
        self.inlet_norm    = (self.inlet - self.v_mean) / (self.v_std + 1e-8)

        self.arc_positions = torch.linspace(0, 1, 256)
        self.k_min = k_min
        self.k_max = k_max
        self.random_taps = random_taps

    def __len__(self):
        return len(self.pressure)

    def __getitem__(self, idx):
        p_norm = self.pressure_norm[idx]
        inlet  = self.inlet_norm[idx]

        k = torch.randint(self.k_min, self.k_max + 1, ()).item()

        if self.random_taps:
            tap_indices = torch.randperm(256)[:k].sort().values
        else:
            tap_indices = torch.arange(0, 256, 256 // k)[:k]

        arc_vals = self.arc_positions[tap_indices]
        p_vals   = p_norm[tap_indices]
        taps     = torch.stack([arc_vals, p_vals], dim=-1)

        inlet_feat = make_inlet_features(inlet.unsqueeze(0)).squeeze(0)

        return {'p_full': p_norm, 'taps': taps, 'inlet': inlet_feat}


def collate_fn(batch):
    max_k = max(item['taps'].shape[0] for item in batch)
    p_list, taps_list, inlet_list, k_list = [], [], [], []
    for item in batch:
        p_list.append(item['p_full'])
        inlet_list.append(item['inlet'])
        k = item['taps'].shape[0]
        k_list.append(k)
        pad = max_k - k
        taps_list.append(torch.cat([item['taps'], torch.zeros(pad, 2)], dim=0))
    return {
        'p_full': torch.stack(p_list),
        'taps'  : torch.stack(taps_list),
        'inlet' : torch.stack(inlet_list),
        'k'     : torch.tensor(k_list, dtype=torch.long),
    }