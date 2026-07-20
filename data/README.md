# Data directory

AccelTraffic uses the two standard public benchmarks, **METR-LA** and
**PEMS-BAY**, released by the DCRNN authors. The raw data is **not redistributed**
here — you download it from DCRNN and generate the acceleration channels with the
preprocessing script in this repository.

```
data/
├── metr-la/      # (created after preprocessing)
│   ├── scaled_speed.npy            # train-only per-sensor z-score speed   (34,272 × 207)
│   ├── scaled_acceleration.npy     # causal-SG-filtered + normalized accel (34,272 × 207)
│   ├── adj_mx.pkl                  # adjacency matrix (from DCRNN)         (207 × 207)
│   └── normalization_params.json   # per-sensor train-only statistics
└── pems-bay/     # same four files (52,116 × 325)
```

## Step 1 — download the raw DCRNN data

From the [DCRNN data folder](https://github.com/liyaguang/DCRNN) (Google Drive
link in that repo), download and place:

```
data/metr-la/metr-la.h5      data/metr-la/adj_mx.pkl
data/pems-bay/pems-bay.h5    data/pems-bay/adj_mx.pkl
```

## Step 2 — generate the acceleration channels

```bash
python preprocessing/generate_acceleration.py --dataset metr-la
python preprocessing/generate_acceleration.py --dataset pems-bay
```

This (i) computes acceleration as the backward finite difference of speed,
(ii) applies the strictly causal Savitzky–Golay filter (W=13, p=1), and
(iii) writes per-sensor **train-only** z-score statistics (computed on the first
70% of the chronological data and applied to all splits — the protocol used in
the paper). It produces `scaled_speed.npy`, `scaled_acceleration.npy`, and
`normalization_params.json`.

## Data format

| File | Shape | Notes |
|---|---|---|
| `scaled_speed.npy` | (timesteps, nodes) | per-sensor z-score (train-only), float32 |
| `scaled_acceleration.npy` | (timesteps, nodes) | `a[t]=(v[t]−v[t−1])/300s` → causal SG (W=13,p=1) → per-sensor z-score (train-only) |
| `adj_mx.pkl` | — | DCRNN tuple `(sensor_ids, sensor_id_to_ind, adj_mx)` |
| `normalization_params.json` | — | per-sensor mean/std (train-only); reference summary values included |

## Preprocessing details

- **Causal SG filter:** W=13, p=1, strictly backward-looking (no future-data
  leakage); selected by grid search over W ∈ {7,…,17}, p ∈ {1,2,3}.
- **Normalization:** per-sensor z-score, statistics computed **from the training
  split only** (first 70%), matching the paper.

## Citations

```bibtex
@article{abahussen2026acceleration,
  title   = {Acceleration-Enriched Input Preprocessing for Traffic Speed
             Forecasting: A Dual-Channel Framework},
  author  = {Aba Hussen, Omar S. and Hashim, Shaiful J. and
             Samsudin, Khairulmizam and Shafri, Helmi Z. M.},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  issn    = {1558-0016},
  doi     = {10.1109/TITS.2026.3708746},
  note    = {Early Access}
}

@inproceedings{li2018dcrnn,
  title     = {Diffusion Convolutional Recurrent Neural Network: Data-Driven
               Traffic Forecasting},
  author    = {Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle = {ICLR},
  year      = {2018}
}
```
