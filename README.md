# CFM Airfoil Pressure Reconstruction
## Uncertainty-Aware Surface Pressure Field Reconstruction from Sparse Taps

---

## Project Summary

Reconstructs full 2D airfoil surface pressure distributions from k sparse pressure
tap measurements using Conditional Flow Matching (CFM). Rather than producing a
single point estimate, the model generates an ensemble of plausible pressure fields,
providing calibrated uncertainty bounds on both the pressure field and the derived
aerodynamic coefficients CL and CD.

**Dataset:** AirfRANS (800 train / 200 test RANS simulations, 2D airfoils)
**Best model:** val loss 0.0527, MAE(p) = 163 Pa at k=10 (p_std = 1330 Pa)

---

## Repository Structure

```
CFM_Airfoil/
├── data/
│   └── preprocess.py        # AirfRANS VTP loading, arc-length interpolation, CL/CD
├── model/
│   ├── encoder.py           # DeepSets encoder with learned positional attention
│   ├── cfm.py               # Conv UNet vector field + CFM loss
│   └── integrator.py        # Euler ODE integration, CL/CD from pressure samples
├── processed/               # Preprocessed train/test/stats NPZ files
├── checkpoints/             # best.pt + epoch checkpoints
├── results/                 # Evaluation outputs and figures
├── dataset.py               # AirfoilDataset, collate_fn
├── train.py                 # Training loop
├── evaluate.py              # MAE / CL / CD / coverage metrics
└── visualize.py             # Pressure field plots, CL/CD posteriors, calibration
```

---

## Setup and Usage

**1. Install dependencies**
```bash
# Install PyTorch first — match your CUDA version (check with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

**2. Download the AirfRANS dataset**

The dataset is downloaded via the `airfrans` Python package, which pulls it from Zenodo (~4 GB).
Run this once from the project root:

```python
import airfrans as af

af.dataset.download(root='./data/airfrans_data', file_name='Dataset', unzip=True)
```

This creates the following structure (1001 simulation directories):
```
data/airfrans_data/
├── Dataset.zip
└── Dataset/
    └── airFoil2D_SST_<Re>_<AoA>_<thickness>_<camber>_<twist>/
        ├── <name>_aerofoil.vtp      ← surface mesh with pressure + normals
        ├── <name>_freestream.vtp    ← inlet velocity [vx, vy]
        └── <name>_internal.vtu      ← volume mesh (not used)
```

**3. Preprocess, train, evaluate**

```bash
# Preprocess AirfRANS dataset
python data/preprocess.py \
    --data_root data/airfrans_data/Dataset/ \
    --output_dir ./processed

# Train — fixed uniform taps (upper bound on reconstruction quality)
python train.py \
    --data_dir ./processed \
    --checkpoint_dir ./checkpoints_uniform \
    --epochs 400 --batch_size 32 --lr 3e-4

# Train — random taps (realistic deployment scenario)
python train.py \
    --data_dir ./processed \
    --checkpoint_dir ./checkpoints_random \
    --epochs 400 --batch_size 32 --lr 3e-4 \
    --random_taps

# Evaluate (pass --random_taps to match the training tap strategy)
python evaluate.py --checkpoint ./checkpoints_uniform/best.pt --k 5 10 20
python evaluate.py --checkpoint ./checkpoints_random/best.pt  --k 5 10 20 --random_taps

# Visualise
python visualize.py --checkpoint ./checkpoints_uniform/best.pt --case_idx 5
```

---

## Architecture

### Overview

The model is a **Conditional Flow Matching** generative model. At inference it
integrates an ODE from Gaussian noise to a pressure field sample, conditioned on
the sparse tap observations. Running this n_samples times produces a posterior
ensemble over the full pressure distribution.

```
taps (k, 2) ──► SparseEncoder ──► c (128,)  ─────────────────────────────────┐
inlet (3,)  ──►                                                               │
                                                                               ▼
x_0 ~ N(0,I) ──► [Euler ODE: dx/dt = v_θ(x_t, t, c)] ──► x_1 (256,) pressure sample
```

### Encoder — `model/encoder.py`

A **DeepSets architecture with learned positional attention**. Tap positions are
randomly sampled during training (k taps drawn uniformly from the 256 surface
points without replacement), so the model must generalise to arbitrary sensor
placements — matching real-world scenarios where taps are constrained by access,
cost, or legacy hardware.

Each tap is a 2-vector `(arc_position, normalised_pressure)` processed by a
shared 3-layer MLP. Taps are aggregated via **softmax attention**, where the
attention weight is a learned function of arc position. With random placements
the model cannot assume a fixed positional pattern, so the attention mechanism
is meaningful — it learns to upweight taps that land near high-information
regions (suction peak, trailing edge) and downweight taps on the flat pressure
side, regardless of how many or where they fall.

After aggregation, the pooled tap embedding is concatenated with:
- inlet velocity features `[vx, vy, AoA]`
- normalised tap count k/k_max (allows the vector field to learn to sharpen its
  posterior when more taps are available)

and projected to a 128-dim condition vector c.

**Why DeepSets over a conv encoder:** With only k=5–20 taps on a 256-point grid,
the input is 92–98% zeros when scattered onto the grid. A convolutional encoder
operating on this sparse signal learns to detect isolated impulses rather than
spatial structure — it reduces to a worse version of DeepSets. The conv inductive
bias only helps when the input is spatially dense, which the vector field input is.

### Vector Field — `model/cfm.py`

A **1D conv UNet** operating on the full 256-point pressure signal:

```
x_t (256,) ──► in_proj ──► [enc0 (256)] ──► down ──► [enc1 (128)] ──► down ──► [mid (64)]
                                                                                      │
out_proj ◄── [dec0 (256)] ◄── up+skip ◄── [dec1 (128)] ◄── up+skip ◄─────────────────┘
```

- 3 resolution levels: 256 → 128 → 64 (bottleneck)
- Each level has n_blocks=2 **FiLM-conditioned residual conv blocks**
- FiLM conditioning: `h = h + act(GroupNorm(conv(h)) * (1 + scale) + shift)`
  where scale and shift are linear projections of the fused condition `[c, t_emb]`
- Circular padding throughout (arc=0 and arc=1 are the same physical point —
  the trailing edge — so the surface is topologically a closed loop)
- Time embedding: 32-dim Fourier embedding of t ∈ [0,1]

**Why a conv UNet over a flat MLP:** The pressure field has strong spatial
correlations — the suction peak at arc≈0.05 influences the recovery over the
next 20% of arc length, and the trailing-edge pressure must be consistent with
the Kutta condition. A flat MLP treats all 256 points as independent. The UNet
encodes these spatial dependencies explicitly via receptive fields that grow with
depth, and the skip connections preserve fine-scale pressure detail (leading-edge
peak, mid-chord inflection) that would otherwise be lost at the bottleneck.

**Why FiLM at every layer:** In early experiments with concatenation-only
conditioning, the network learned to ignore the condition vector and collapsed to
predicting the marginal pressure distribution (val loss ~0.36). FiLM injects
the condition as a multiplicative and additive modulation at every spatial scale,
making it structurally impossible for the network to route around the condition.

### Training

- **Loss:** CFM straight-line OT path — `MSE(v_θ(x_t, t, c), x_1 - (1-σ_min)x_0)`
- **Tap sampling:** k ~ Uniform[5, 20], tap positions drawn randomly without replacement from the 256 surface points — sorted by arc position before passing to the encoder
- **Normalisation:** Pressure standardised to zero mean, unit variance using
  training set statistics. Inlet velocity similarly normalised.
- **Optimiser:** AdamW, lr=3e-4, weight_decay=1e-4
- **Schedule:** CosineAnnealingLR over 400 epochs
- **Gradient clipping:** max norm 1.0
- **Total parameters:** ~1.9M

---

## Key Fixes Made During Development

### 1. CL/CD Integration Bug (preprocess.py, integrator.py)
The original code computed `np.trapezoid(p * n * ds)` which double-applies the
arc-length spacing — trapezoid internally computes spacing from the x-axis, but
ds was already multiplied in. Fixed to `np.trapezoid(p * n, x=arc)` where arc is
the cumulative arc-length coordinate.

### 2. Posterior Collapse (model/cfm.py)
Early architecture used a flat MLP with condition concatenated only at the input
layer (val loss ~0.36, MAE ~250 Pa). The network learned to ignore the condition.
Fixed by switching to FiLM conditioning injected at every layer of the UNet,
forcing the condition to modulate activations at every spatial scale.

### 3. Train/Test Distribution Mismatch (data/preprocess.py)
A random 80/20 split resulted in the test set containing ~1.7x more stalled/
high-CD cases than the train set (CD>0.10: 15.5% test vs 9.0% train). This
inflated val loss and caused noisy checkpointing. Fixed with a stratified split
by CD decile.

### 4. Sparse Encoder Scatter Bug (model/encoder.py)
The vectorised scatter operation routed padded (zero) taps to index 0, silently
overwriting any real tap at arc=0 (the trailing edge stagnation point — the
highest-weight position). Fixed by routing padded taps to a dummy slot outside
the valid range.

### 5. k-Aware Conditioning (model/encoder.py)
With uniform taps the model had no way to know whether k=5 or k=20 taps were
provided, so it could not learn to sharpen its posterior with more information.
Fixed by appending normalised tap count k/k_max to the fusion layer input.

---

## Tap Placement Ablation

Two models are trained with identical architecture, data, and hyperparameters —
differing only in tap placement strategy:

| Strategy | Description | Use case |
|---|---|---|
| **Uniform** (`--random_taps` off) | Taps at evenly-spaced arc positions 0, 1/k, 2/k, … | Optimised sensor placement; upper bound on quality |
| **Random** (`--random_taps`) | k taps drawn uniformly at random from 256 surface points | Arbitrary/legacy sensor arrays; realistic deployment |

Under random placement, the learned attention in the encoder is meaningful —
the model must discover which positions are informative from the pressure values
alone, since it cannot assume a fixed pattern. Under uniform placement, attention
weights would converge to constants and mean pooling is equivalent.

## Evaluation Results

### Overall (100 test cases, 100 samples per case)

| k | MAE(p) | MAE(CL) | MAE(CD) | CL cov% | CD cov% |
|---|--------|---------|---------|---------|---------|
| 5 | 195 Pa | 0.106 | 0.015 | 94% | 96% |
| 10 | 163 Pa | 0.082 | 0.012 | 99% | 98% |
| 20 | 169 Pa | 0.081 | 0.012 | 99% | 98% |

Coverage target: 95% for ±2σ intervals. k=5 is well-calibrated; k=10/20 are
conservative (over-dispersed posterior — uncertainty bands are wider than needed).

### MAE(p) by Angle of Attack (k=10)

| AoA range | n cases | Mean MAE | 90th pct MAE |
|-----------|---------|----------|--------------|
| -5° to 0° | 54 | 219 Pa | 553 Pa |
| 0° to +5° | 60 | 137 Pa | 261 Pa |
| +5° to +10° | 40 | 102 Pa | 164 Pa |
| +10° to +15° | 46 | 109 Pa | 210 Pa |

**Negative AoA cases are significantly harder.** The 10 worst-performing cases
are all at AoA between -2° and -5°. At negative AoA the pressure side becomes
the suction surface, producing a suction peak on the lower surface that is poorly
captured by the leading-edge-start arc-length ordering. The model effectively
sees the suction peak in an unfamiliar arc-length position for these cases.

### MAE(p) by CD (k=10)

| CD range | n cases | Mean MAE |
|----------|---------|----------|
| 0.00 – 0.01 (attached) | 119 | 150 Pa |
| 0.01 – 0.05 (mild sep.) | 30 | 183 Pa |
| 0.05 – 0.10 (separated) | 15 | 99 Pa |
| 0.10 – 0.25 (stalled) | 31 | 119 Pa |

Counterintuitively, stalled cases (CD > 0.05) are *easier* to reconstruct than
mildly separated cases. Stalled airfoils have a broadly flat low-pressure region
that is easier to interpolate from sparse taps; mildly separated cases have
sharp local features that are sensitive to exact tap placement.

### Valid Operating Range

The model is well-validated for:
- AoA: 0° to +15° (mean MAE 102–137 Pa)
- CL: 0.5 to 2.0 (mean MAE 115–120 Pa)
- CD: 0.00 to 0.25 across the full trained range

Use with caution for:
- AoA: -5° to 0° (mean MAE 219 Pa, 90th pct 553 Pa)
- CL: -1.0 to 0.5 (negative lift / low-lift cases, mean MAE 177–178 Pa)
