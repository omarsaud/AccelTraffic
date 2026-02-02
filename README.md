# AccelTraffic: Acceleration-Driven Traffic Speed Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Reference implementation of **acceleration-driven traffic speed prediction** with causal Savitzky--Golay filtering and dual-channel normalization.

![Conceptual Framework](figures/fig5_conceptual_framework.png)

## Overview

This repository provides a **model-agnostic preprocessing framework** that integrates acceleration as a causally filtered auxiliary channel with dual-channel normalization for traffic speed prediction. The framework is evaluated on four spatiotemporal graph neural network backbones:

- **DCRNN** - Diffusion Convolutional Recurrent Neural Network ([Li et al., ICLR 2018](https://github.com/liyaguang/DCRNN))
- **AGCRN** - Adaptive Graph Convolutional Recurrent Network ([Bai et al., NeurIPS 2020](https://github.com/LeiBAI/AGCRN))
- **STGIN** - Spatial-Temporal Graph Informed Network ([Zou et al.](https://github.com/zouguojian/STGIN))
- **Graph WaveNet** - Graph WaveNet for Deep Spatial-Temporal Graph Modeling ([Wu et al., IJCAI 2019](https://github.com/nnzhan/Graph-WaveNet))

## Key Contributions

1. **Acceleration-aware preprocessing**: Derives acceleration from speed and uses it as an auxiliary input channel (speed-only prediction target)
2. **Strictly causal Savitzky-Golay filtering**: Data-driven parameter selection (W=13, p=1) avoiding future information leakage
3. **Per-sensor dual-channel normalization**: Independent normalization for speed and acceleration channels
4. **Architecture-agnostic evaluation**: Consistent MAE improvements across all four backbones

## Results Summary (Q=12, 60-min horizon)

### METR-LA Dataset

| Model | NoAcc (Baseline) | Acc_SG (Proposed) | Improvement |
| ----- | ---------------- | ----------------- | ----------- |
| AGCRN | 5.05             | 4.27              | -15.4%      |
| DCRNN | 5.31             | 4.44              | -16.4%      |
| GWNET | 4.66             | 4.27              | -8.4%       |
| STGIN | 5.48             | 4.95              | -9.7%       |

### PEMS-BAY Dataset

| Model | NoAcc (Baseline) | Acc_SG (Proposed) | Improvement |
| ----- | ---------------- | ----------------- | ----------- |
| AGCRN | 1.78             | 1.64              | -7.9%       |
| DCRNN | 1.91             | 1.75              | -8.4%       |
| GWNET | 1.84             | 1.69              | -8.2%       |
| STGIN | 2.22             | 1.94              | -10.8%      |

Full results (72 experiments, 4 models × 2 datasets × 3 horizons × 3 configs) are provided as:

- `results/experiments_72.csv` (per-run table)
- `results/metrla_ablation.csv` (Table METR-LA)
- `results/pemsbay_ablation.csv` (Table PEMS-BAY)
- `results/significance_summary.csv` (paired t-test summary)
- `results/sg_sensitivity.csv` (SG window sensitivity)
- `results/timing_table.csv` (training-time overhead)

## Figures

- `figures/fig5_conceptual_framework.png` (conceptual framework)
- `figures/fig6_baseline_mae.png` (baseline MAE vs horizon)
- `figures/fig7_acc_sg_improvement.png` (Acc_SG vs Acc_NoSG improvement)
- `figures/fig8_event_metrla.png` and `figures/fig9_event_pemsbay.png` (case studies)
- `figures/fig11_timing_comparison.png` (training-time overhead)
- `figures/fig12_failure_analysis.png` (free-flow regime analysis)

## Repository Structure

```
AccelTraffic/
├── models/                    # Model architectures (4 models)
│   ├── dcrnn_model.py        # DCRNN
│   ├── agcrn_model.py        # AGCRN
│   ├── gwnet_model.py        # Graph WaveNet
│   ├── stgin_model.py        # STGIN
│   └── model_factory.py      # Model creation utilities
├── preprocessing/             # Data preprocessing
│   ├── simple_data_loading.py    # Multi-model loader (DCRNN/GWNET/AGCRN)
│   ├── stgin_data_loading.py     # STGIN loader (with STE embeddings)
│   ├── generate_acceleration.py  # Acceleration preprocessing
│   └── sg_parameter_search.py    # SG filter parameter optimization
├── utils/                     # Utilities
│   ├── evaluation_utils.py   # Metrics (MAE, RMSE, MAPE)
│   ├── model_optimizer.py    # Training optimizations (AMP, TF32)
│   ├── global_configuration.py # GPU optimizations
│   └── utils_misc.py         # Seed setting utilities
├── data/                      # Data files (download required)
│   ├── metr-la/
│   │   ├── scaled_speed.npy           # Normalized speed
│   │   ├── scaled_acceleration.npy    # SG-filtered + normalized acceleration
│   │   ├── adj_mx.pkl                 # Adjacency matrix
│   │   └── normalization_params.json  # Statistics
│   └── pems-bay/
│       ├── scaled_speed.npy
│       ├── scaled_acceleration.npy
│       ├── adj_mx.pkl
│       └── normalization_params.json
├── results/                   # Experiment results (72 runs)
│   ├── experiments_72.csv             # All 72 experiments
│   ├── metrla_ablation.csv            # METR-LA results
│   ├── pemsbay_ablation.csv           # PEMS-BAY results
│   ├── significance_summary.csv       # Statistical tests
│   ├── timing_table.csv               # Training overhead
│   └── stats_pairwise_tests_4models.csv
├── figures/                   # Publication figures
│   ├── fig5_conceptual_framework.png
│   ├── fig6_baseline_mae.png
│   ├── fig7_acc_sg_improvement.png
│   ├── fig8_event_metrla.png
│   └── fig9_event_pemsbay.png
├── train_multimodel.py       # Train DCRNN, AGCRN, GWNET
├── train_stgin.py            # Train STGIN
├── SETUP.md                  # Installation guide
└── README.md
```

## Data Loaders

AccelTraffic uses **two different data loaders** depending on the model:

### 1. Simple Data Loader (DCRNN, GWNET, AGCRN)

- **File:** `preprocessing/simple_data_loading.py`
- **Usage:** Multi-model training script (`train_multimodel.py`)
- **Features:**
  - Raw speed + acceleration channels
  - No spatiotemporal embeddings (STE)
  - Optimized DataLoader (num_workers, pin_memory, prefetch)
- **Format:** Returns `(batch, nodes, seq_len, channels)`

### 2. STGIN Data Loader (STGIN only)

- **File:** `preprocessing/stgin_data_loading.py`
- **Usage:** STGIN training script (`train_stgin.py`)
- **Features:**
  - Speed + acceleration channels
  - **Spatiotemporal embeddings (STE)** - time/day/spatial encoding
  - STE caching for efficiency
- **Format:** Returns `(x, y, ste)` where `ste` contains temporal embeddings
- **Why different:** STGIN architecture requires explicit time/space encodings

**Note:** The two loaders are intentionally separate because STGIN has a fundamentally different architecture that requires STE embeddings for its attention mechanisms.

## Installation

```bash
git clone https://github.com/omarsaud/AccelTraffic.git
cd AccelTraffic
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- NumPy, Pandas, SciPy, scikit-learn, h5py

## Datasets

We use two standard traffic forecasting benchmarks from the [DCRNN repository](https://github.com/liyaguang/DCRNN):

### METR-LA

- **Sensors**: 207 loop detectors on Los Angeles highways
- **Time Range**: March 2012 - June 2012 (34,272 timesteps)
- **Interval**: 5 minutes
- **Features**: Speed (mph)

### PEMS-BAY

- **Sensors**: 325 sensors in the San Francisco Bay Area
- **Time Range**: January 2017 - May 2017 (52,116 timesteps)
- **Interval**: 5 minutes
- **Features**: Speed (mph)

### Dataset Statistics

| Dataset  | Sensors | Timesteps | Mean speed | Std. speed |
| -------- | ------- | --------- | ---------- | ---------- |
| METR-LA  | 207     | 34,272    | 53.7 mph   | 20.3 mph   |
| PEMS-BAY | 325     | 52,116    | 62.6 mph   | 9.6 mph    |

### Data Download

**IMPORTANT:** Data files are **not included** in this repository due to size limitations.

**Option 1: Download Preprocessed (Recommended)**
Download preprocessed `.npy` files from [Google Drive](https://drive.google.com/...) and extract to `data/metr-la/` and `data/pems-bay/`.

**Option 2: Generate from Raw Data**

1. Download raw data from [DCRNN Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
2. Place in `data/metr-la/metr-la.h5` and `data/pems-bay/pems-bay.h5`
3. Generate acceleration channels:
   ```bash
   python preprocessing/generate_acceleration.py --dataset metr-la
   python preprocessing/generate_acceleration.py --dataset pems-bay
   ```

**See [`data/README.md`](data/README.md) for detailed instructions.**

## Training

### Train DCRNN, AGCRN, or Graph WaveNet

```bash
# Train with acceleration (Acc_SG - uses preprocessed SG-filtered data)
python train_multimodel.py --model agcrn --dataset metr-la --Q 12 --use_acceleration true --epochs 100

# Train without acceleration (NoAcc baseline)
python train_multimodel.py --model agcrn --dataset metr-la --Q 12 --use_acceleration false --epochs 100
```

**Available models:** `dcrnn`, `gwnet`, `agcrn`
**Available datasets:** `metr-la`, `pems-bay`
**Prediction horizons (Q):** `3` (15 min), `6` (30 min), `12` (60 min)

### Train STGIN

```bash
python train_stgin.py --Q 12 --use_acceleration true --batch_size 32 --epochs 100
```

### Configuration Options

| Config     | Channels | Command Flag                 | Description                                          |
| ---------- | -------- | ---------------------------- | ---------------------------------------------------- |
| `NoAcc`  | 1        | `--use_acceleration false` | Speed-only baseline                                  |
| `Acc_SG` | 2        | `--use_acceleration true`  | Speed + causally SG-filtered acceleration (proposed) |

**Note:** SG filtering is applied during preprocessing. Using `--use_acceleration true` automatically loads the preprocessed SG-filtered acceleration data from `data/<dataset>/scaled_acceleration.npy`.

## Preprocessing Framework

### 1. Acceleration Derivation

Acceleration is computed as the temporal derivative of speed:

```python
acceleration[t] = (speed[t] - speed[t-1]) / dt  # dt = 300 seconds
```

**Critical**: Acceleration serves exclusively as an auxiliary input feature. The prediction target is speed only.

### 2. Causal Savitzky-Golay Filter

The framework applies a **strictly causal** SG filter (W=13, p=1) to raw acceleration using only past and present samples:

```python
from scipy.signal import savgol_filter
import numpy as np

def causal_sg_filter(signal, window_length=13, polyorder=1):
    # Pad future with edge values to make filter causal
    padded = np.pad(signal, (0, window_length-1), mode='edge')
    filtered = savgol_filter(padded, window_length, polyorder)
    return filtered[:len(signal)]  # Truncate to original length
```

### SG Parameter Selection

To find optimal SG parameters for your dataset:

```bash
python preprocessing/sg_parameter_search.py --dataset metr-la --horizon 12
```

**Selection Criterion**: Maximize correlation of filtered acceleration with future speed (predictive power).

| Parameter      | Search Range     | Optimal      |
| -------------- | ---------------- | ------------ |
| Window (W)     | 7, 9, 11, 13, 15 | **13** |
| Polynomial (p) | 1, 2, 3          | **1**  |

Parameter selection is performed on training data to ensure fair evaluation.

### 3. Dual-Channel Normalization

Per-sensor z-score normalization is applied independently to each channel:

**METR-LA (from `data/metr_la_normalization.json`):**

| Channel      | Mean (μ) | Std (σ) | Unit  |
| ------------ | --------- | -------- | ----- |
| Speed        | 53.72     | 20.26    | mph   |
| Acceleration | ≈0       | 0.0086   | m/s² |

**PEMS-BAY (from `data/pems_bay_normalization.json`):**

| Channel      | Mean (μ) | Std (σ) | Unit  |
| ------------ | --------- | -------- | ----- |
| Speed        | 62.62     | 9.59     | mph   |
| Acceleration | ≈0       | 0.0034   | m/s² |

## Evaluation

### Metrics

- **MAE** (Mean Absolute Error): Primary metric (mph)
- **RMSE** (Root Mean Squared Error): Sensitivity to large errors
- **MAPE** (Mean Absolute Percentage Error): Relative error (%)

### Prediction Horizons

- **Q=3**: 15 minutes ahead
- **Q=6**: 30 minutes ahead
- **Q=12**: 60 minutes ahead

### Statistical Significance

All comparisons tested with paired t-tests. See:

- `results/significance_summary.csv` (summary table)
- `results/stats_pairwise_tests_4models.csv` (full paired-test export for the 4 models)

## Citation

If you find this work useful, please cite the associated paper:

```bibtex
@article{abahussen2025acceleration,
  title={Acceleration-Driven Deep Learning for Traffic Speed Prediction with Causal Filtering and Dual-Channel Normalization},
  author={Aba Hussen, Omar S. and Hashim, Shaiful J. and Samsudin, Khairulmizam and Shafri, Helmi Z. M.},
  year={2025}
}
```

## Acknowledgments

This work builds upon the following open-source implementations:

- [DCRNN](https://github.com/liyaguang/DCRNN) - Li et al., ICLR 2018
- [AGCRN](https://github.com/LeiBAI/AGCRN) - Bai et al., NeurIPS 2020
- [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) - Wu et al., IJCAI 2019
- [STGIN](https://github.com/zouguojian/STGIN) - Zou et al.

Datasets (METR-LA and PEMS-BAY) are provided by the DCRNN authors.

## Contact

- **Omar S. Aba Hussen** - Omarabahussen@gmail.com
- **Shaiful J. Hashim** - sjh@upm.edu.my

Department of Computer and Communication Systems, Faculty of Engineering, Universiti Putra Malaysia (UPM)
