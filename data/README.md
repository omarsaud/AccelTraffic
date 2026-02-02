# AccelTraffic Data Directory

This directory contains preprocessed traffic speed and acceleration data for two benchmark datasets.

## Directory Structure

```
data/
├── metr-la/
│   ├── scaled_speed.npy           # Normalized speed data (34,272 × 207)
│   ├── scaled_acceleration.npy     # SG-filtered + normalized acceleration (34,272 × 207)
│   ├── adj_mx.pkl                  # Adjacency matrix (207 × 207)
│   ├── normalization_params.json   # Speed/acceleration mean/std
│   └── graph_sensor_locations_METR-LA.csv  # Sensor coordinates
│
└── pems-bay/
    ├── scaled_speed.npy           # Normalized speed data (52,116 × 325)
    ├── scaled_acceleration.npy     # SG-filtered + normalized acceleration (52,116 × 325)
    ├── adj_mx.pkl                  # Adjacency matrix (325 × 325)
    └── normalization_params.json   # Speed/acceleration mean/std
```

## Data Not Included in Repository

Due to file size limitations, the actual data files (`.npy`, `.pkl`) are **not included** in this GitHub repository. You must download or generate them separately.

## Option 1: Download Preprocessed Data (Recommended)

Download from Google Drive (if available):
- [METR-LA preprocessed](https://drive.google.com/...) 
- [PEMS-BAY preprocessed](https://drive.google.com/...)

Extract to the appropriate folders:
```bash
# Extract METR-LA
unzip metr-la-preprocessed.zip -d data/metr-la/

# Extract PEMS-BAY
unzip pems-bay-preprocessed.zip -d data/pems-bay/
```

## Option 2: Download Raw Data and Generate

### Step 1: Download Raw Data from DCRNN Repository

Download the original datasets from [DCRNN Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX):

- `metr-la.h5` (speed data)
- `adj_mx_metr.pkl` (adjacency matrix)
- `pems-bay.h5` (speed data)
- `adj_mx_bay.pkl` (adjacency matrix)

Place raw files temporarily:
```bash
data/
├── metr-la/
│   ├── metr-la.h5
│   └── adj_mx.pkl
└── pems-bay/
    ├── pems-bay.h5
    └── adj_mx.pkl
```

### Step 2: Generate Preprocessed Data

Run the preprocessing script:
```bash
# Generate METR-LA
python preprocessing/generate_acceleration.py --dataset metr-la

# Generate PEMS-BAY
python preprocessing/generate_acceleration.py --dataset pems-bay
```

This will create:
- `scaled_speed.npy` - Per-sensor z-score normalized speed
- `scaled_acceleration.npy` - Causal SG-filtered (W=13, p=1) + normalized acceleration
- `normalization_params.json` - Speed and acceleration statistics

## Data Format

### scaled_speed.npy
- Shape: `(timesteps, nodes)`
- Type: `float32`
- Normalization: Per-sensor z-score (mean=0, std=1 per sensor across full dataset)
- Units: Normalized mph

### scaled_acceleration.npy
- Shape: `(timesteps, nodes)`
- Type: `float32`
- Preprocessing: 
  1. Compute derivative: `a[t] = (v[t] - v[t-1]) / 300s`
  2. Apply causal SG filter (W=13, p=1)
  3. Per-sensor z-score normalization
- Units: Normalized m/s²

### adj_mx.pkl
- Format: Pickle file with tuple `(sensor_ids, sensor_id_to_ind, adj_mx)`
- `adj_mx`: `(nodes, nodes)` weighted adjacency matrix

### normalization_params.json
- `speed_mean`: Global mean speed (mph) before normalization
- `speed_std`: Global std speed (mph) before normalization
- `accel_mean`: Global mean acceleration (m/s²) before normalization
- `accel_std`: Global std acceleration (m/s²) before normalization

**Note:** These are global statistics for reference. Actual normalization is per-sensor.

## Verification

After downloading/generating, verify files exist:
```bash
python -c "
from pathlib import Path
import numpy as np

for ds in ['metr-la', 'pems-bay']:
    print(f'\n{ds.upper()}:')
    speed = Path(f'data/{ds}/scaled_speed.npy')
    accel = Path(f'data/{ds}/scaled_acceleration.npy')
    adj = Path(f'data/{ds}/adj_mx.pkl')
    norm = Path(f'data/{ds}/normalization_params.json')
    
    print(f'  Speed: {'✅' if speed.exists() else '❌'} {np.load(speed).shape if speed.exists() else 'Missing'}')
    print(f'  Accel: {'✅' if accel.exists() else '❌'} {np.load(accel).shape if accel.exists() else 'Missing'}')
    print(f'  Adj:   {'✅' if adj.exists() else '❌'}')
    print(f'  Norm:  {'✅' if norm.exists() else '❌'}')
"
```

Expected output:
```
METR-LA:
  Speed: ✅ (34272, 207)
  Accel: ✅ (34272, 207)
  Adj:   ✅
  Norm:  ✅

PEMS-BAY:
  Speed: ✅ (52116, 325)
  Accel: ✅ (52116, 325)
  Adj:   ✅
  Norm:  ✅
```

## Preprocessing Details

### Savitzky-Golay Filter Parameters
- **Window (W)**: 13 timesteps (65 minutes)
- **Polynomial order (p)**: 1 (linear)
- **Causality**: Strictly backward-looking (no future data leakage)

Selected via grid search to maximize correlation with future speed while maintaining causality.

### Normalization
- **Method**: Per-sensor z-score
- **Formula**: `x_norm = (x - mean) / std`
- **Statistics**: Computed over full dataset (not train-only)

This follows the methodology in the AccelTraffic paper.

## Citation

If you use this preprocessed data, please cite:

```bibtex
@article{abahussen2025acceleration,
  title={Acceleration-Driven Deep Learning for Traffic Speed Prediction with Causal Filtering and Dual-Channel Normalization},
  author={Aba Hussen, Omar S. and Hashim, Shaiful J. and Samsudin, Khairulmizam and Shafri, Helmi Z. M.},
  year={2025}
}
```

## Original Data Citation

METR-LA and PEMS-BAY datasets are from:

```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={ICLR},
  year={2018}
}
```
