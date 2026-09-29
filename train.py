"""
train.py
Run:
    python train.py --data_dir ./processed --epochs 300 --batch_size 32
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.encoder import SparseEncoder, make_inlet_features
from model.cfm import VectorField, cfm_loss
from dataset import AirfoilDataset, collate_fn


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = AirfoilDataset(
        os.path.join(args.data_dir, 'train.npz'),
        os.path.join(args.data_dir, 'stats.npz'),
        k_min=args.k_min, k_max=args.k_max,
        random_taps=args.random_taps,
    )
    val_ds = AirfoilDataset(
        os.path.join(args.data_dir, 'test.npz'),
        os.path.join(args.data_dir, 'stats.npz'),
        k_min=args.k_min, k_max=args.k_max,
        random_taps=args.random_taps,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    encoder = SparseEncoder(
        tap_input_dim=2, inlet_dim=3,
        hidden_dim=128, cond_dim=128,
    ).to(device)

    vf = VectorField(
        n_surface=256, cond_dim=128,
        time_dim=32, base_ch=64, n_blocks=2,
    ).to(device)

    n_params = sum(p.numel() for p in
                   list(encoder.parameters()) + list(vf.parameters()))
    print(f"Total parameters: {n_params:,}")

    params    = list(encoder.parameters()) + list(vf.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        encoder.train(); vf.train()
        train_losses = []
        for batch in tqdm(train_loader,
                          desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            p_full = batch['p_full'].to(device)
            taps   = batch['taps'].to(device)
            inlet  = batch['inlet'].to(device)
            k      = batch['k'].to(device)

            loss = cfm_loss(vf, encoder, p_full, taps, inlet, k)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        encoder.eval(); vf.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                p_full = batch['p_full'].to(device)
                taps   = batch['taps'].to(device)
                inlet  = batch['inlet'].to(device)
                k      = batch['k'].to(device)
                val_losses.append(
                    cfm_loss(vf, encoder, p_full, taps, inlet, k).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch'   : epoch,
                'encoder' : encoder.state_dict(),
                'vf'      : vf.state_dict(),
                'val_loss': val_loss,
                'args'    : vars(args),
            }, os.path.join(args.checkpoint_dir, 'best.pt'))
            print(f"  ✓ Saved best (val {val_loss:.4f})")

        if epoch % 50 == 0:
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'vf': vf.state_dict()},
                       os.path.join(args.checkpoint_dir, f'epoch_{epoch:03d}.pt'))

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       type=str,   default='./processed')
    parser.add_argument('--checkpoint_dir', type=str,   default='./checkpoints')
    parser.add_argument('--epochs',         type=int,   default=300)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--k_min',          type=int,   default=5)
    parser.add_argument('--k_max',          type=int,   default=20)
    parser.add_argument('--random_taps',    action='store_true', default=False)
    args = parser.parse_args()
    train(args)