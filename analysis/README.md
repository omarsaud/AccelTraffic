# Analysis scripts

These scripts reproduce the analyses reported in the paper from trained model
outputs. They are **post-hoc**: first train the models (see the top-level
`README.md`), then run the analysis you want. Each script writes its table into
the repository's `results/` folder.

| Script | Reproduces | Reads | Writes (`results/`) |
|---|---|---|---|
| `diebold_mariano_test.py` | Diebold–Mariano significance (Table VI) | `models/<run>/predictions_mph.npy`, `targets_mph.npy` | `07_dm_test_results.csv` |
| `regime_breakdown.py` | Per-regime MAE (Table IV) | `models/<run>/predictions_mph.npy`, `targets_mph.npy` | `regime_breakdown.csv` |
| `ctm_baseline.py` | CTM/LWR physical baseline (Table III) | `data/<dataset>/` | `ctm_baseline_results.csv` |
| `mutual_information.py` | Mutual information + Granger (Fig. 6) | `data/<dataset>/scaled_*.npy` | `mutual_information_results.csv`, `granger_causality_results.csv` |
| `granger_causality.py` | Per-sensor Granger over all sensors (Fig. 6B) | `data/<dataset>/scaled_*.npy` | `granger_causality_allsensors.csv` |
| `computational_overhead.py` | Preprocessing/inference cost (Sec. VII-F) | `models/<run>/` | `computational_overhead.csv` |
| `ablation_summary.py` | Component ablations (Table VII) | control-ablation run folders | ablation summary CSVs |

## Run layout

The scripts expect trained runs under `models/` using the naming convention
produced by the training scripts:

```
models/<model>_<dataset>_<config>_LSTM_Q<Q>/
    ├── predictions_mph.npy
    ├── targets_mph.npy
    └── test_results.json
```

for example `models/agcrn_metr-la_Acc_SG_LSTM_Q12/`. The control ablations
(`SpeedSpeed`, `UnifiedNorm`, `AccelOnly`) follow the same convention with the
corresponding config token.

## Example

```bash
# after training the relevant runs and downloading the data:
python analysis/diebold_mariano_test.py     # significance (Table VI)
python analysis/regime_breakdown.py         # per-regime MAE (Table IV)
python analysis/mutual_information.py        # MI + Granger (Fig. 6)
```

## Notes

- These reproduce what the paper reports; the result tables shipped at the top
  of `results/` (`main_results.csv`, `diebold_mariano.csv`, …) are the
  published values and are provided so the repository is usable without
  retraining.
- Datasets are not bundled (see the top-level `README.md` for download); scripts
  that read `data/<dataset>/` require the preprocessed `.npy` files in place.
