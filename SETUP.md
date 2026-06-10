# Setup guide

Step-by-step instructions to install AccelTraffic, prepare the data, train a
model, and reproduce the paper's analyses.

## 1. Install

```bash
git clone https://github.com/omarsaud/AccelTraffic.git
cd AccelTraffic
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

Requirements: Python ≥ 3.8, PyTorch ≥ 2.0 (CUDA optional but recommended), plus
NumPy, Pandas, SciPy, scikit-learn, h5py, matplotlib, tqdm.

Quick check:

```bash
python -c "import torch, models; print('torch', torch.__version__); print('models:', models.__all__)"
```

## 2. Prepare the data

The benchmarks are the standard public DCRNN datasets and are **not
redistributed** here. Download `metr-la.h5` / `pems-bay.h5` and the adjacency
matrices from the [DCRNN data folder](https://github.com/liyaguang/DCRNN), place
them under `data/<dataset>/`, then generate the acceleration channels:

```bash
python preprocessing/generate_acceleration.py --dataset metr-la
python preprocessing/generate_acceleration.py --dataset pems-bay
```

This computes the backward finite-difference acceleration, applies the strictly
causal Savitzky–Golay filter (W=13, p=1), and writes the per-sensor **train-only**
normalization (matching the paper). See [`data/README.md`](data/README.md).

## 3. Train

```bash
# one run (AGCRN, METR-LA, 60-min, proposed Acc_SG)
python train_multimodel.py --model agcrn --dataset metr-la --Q 12 --use_acceleration true --epochs 100

# speed-only baseline for the same cell
python train_multimodel.py --model agcrn --dataset metr-la --Q 12 --use_acceleration false --epochs 100

# transformer backbone
python train_multimodel.py --model staeformer --dataset pems-bay --Q 12 --use_acceleration true --epochs 100

# STGIN uses its own script (spatiotemporal embeddings)
python train_stgin.py --Q 12 --use_acceleration true --epochs 100
```

Each run writes to `models/<model>_<dataset>_<config>_LSTM_Q<Q>/`
(`predictions_mph.npy`, `targets_mph.npy`, `test_results.json`, `best_model.pt`).

## 4. Reproduce the analyses

After the relevant runs exist:

```bash
python analysis/diebold_mariano_test.py     # significance (Table VI)
python analysis/regime_breakdown.py         # per-regime MAE (Table IV)
python analysis/ctm_baseline.py             # CTM/LWR baseline (Table III)
python analysis/mutual_information.py        # MI + Granger (Fig. 6)
python analysis/computational_overhead.py    # deployment cost
```

Outputs are written to `results/`. The published result tables are already
provided there (`main_results.csv`, `diebold_mariano.csv`, …) so the repository
is usable without retraining. See [`analysis/README.md`](analysis/README.md).

## Troubleshooting

- **`ModuleNotFoundError: models`** — run scripts from the repository root.
- **CUDA out of memory** — lower `--batch_size` (PEMS-BAY at Q=12 is the heaviest).
- **Reproducibility** — all runs use a fixed seed (42); minor hardware/library
  differences can still cause small (<0.05 mph) deviations.
