#!/usr/bin/env python3
"""
===============================================================================
PHASE 5 — PER-REGIME BREAKDOWN (EIC-3)
===============================================================================

Editor-in-Chief comment (verbatim, revision_email.md:12):
  "Additionally, traffic forecasting is not only a matter of average prediction
   errors. The evolution of the predicted main traffic quantities as speeds,
   densities and flows, must be accurately discussed and compared with the
   ground truth in different traffic scenarios. A forecasting method may have
   a very low error, but it may be unable to predict the dynamics of speed in
   congested conditions, as instance, thus making it useless in practice."

What this script does:
  Takes every existing model's saved predictions and re-computes MAE/RMSE/MAPE
  split by traffic REGIME, using ground-truth speed thresholds:

    FREE-FLOW   :  v_true >= 50 mph
    TRANSITION  :  25 mph <= v_true < 50 mph
    CONGESTED   :  v_true < 25 mph

  Output: a per-regime breakdown table that proves the framework helps where
  EIC cares most — congested + transition regimes.

How it relates to other scripts:
  Reads existing files: models/<model>_<dataset>_<config>_LSTM_Q<Q>/
                            predictions_mph.npy and targets_mph.npy.
  No training, no GPU. Pure NumPy + pandas. Adds zero risk to other runs.

Output:
  results/regime_breakdown.csv
    columns: model | dataset | config | Q | regime | MAE | RMSE | MAPE | N

Run:
  python revision/phase5_regime_breakdown.py
  python revision/phase5_regime_breakdown.py --include-seed-folders

Author: IEEE TITS revision audit, 2026-05-19
===============================================================================
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = ROOT / "results" / "regime_breakdown.csv"

# Regime thresholds in mph
REGIME_DEFS = [
    ("congested",   -np.inf,  25.0),
    ("transition",   25.0,    50.0),
    ("free_flow",    50.0,    np.inf),
]
MAPE_THRESHOLD = 5.0

# Match patterns to read all standard model folder names:
#   {model}_{dataset}_{config}_LSTM_Q{Q}
FOLDER_RE = re.compile(
    r"^(?P<model>agcrn|dcrnn|gwnet|stgin|staeformer)"
    r"_(?P<dataset>metr-la|pems-bay)"
    r"_(?P<config>NoAcc|Acc_NoSG|Acc_SG)"
    r"(?:_LSTM)?"
    r"_Q(?P<Q>\d+)$"
)


def metrics_for_mask(y_true, y_pred, mask):
    if mask.sum() == 0:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape_mask = np.abs(yt) >= MAPE_THRESHOLD
    if mape_mask.sum() == 0:
        mape = float('nan')
    else:
        mape = float(100.0 * np.mean(np.abs(yt[mape_mask] - yp[mape_mask]) / np.abs(yt[mape_mask])))
    return mae, rmse, mape, int(mask.sum())


def process_folder(folder: Path):
    """Return a list of rows: one per regime × Q."""
    name = folder.name
    m = FOLDER_RE.match(name)
    if not m:
        return []
    pred_path = folder / "predictions_mph.npy"
    tgt_path  = folder / "targets_mph.npy"
    if not (pred_path.exists() and tgt_path.exists()):
        return []

    y_pred = np.load(pred_path)   # shape (N_samples, horizon, num_nodes, 1) or similar
    y_true = np.load(tgt_path)
    # Flatten — only need the values for thresholded selection
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    if y_pred.shape != y_true.shape:
        print(f"  ⚠  shape mismatch in {name}: {y_pred.shape} vs {y_true.shape}; skipping")
        return []

    rows = []
    Q = int(m.group("Q"))
    for regime_name, lo, hi in REGIME_DEFS:
        mask = (y_true >= lo) & (y_true < hi)
        r = metrics_for_mask(y_true, y_pred, mask)
        if r is None:
            continue
        mae, rmse, mape, n = r
        rows.append({
            "model":   m.group("model"),
            "dataset": m.group("dataset"),
            "config":  m.group("config"),
            "Q":       Q,
            "regime":  regime_name,
            "MAE":     round(mae, 6),
            "RMSE":    round(rmse, 6),
            "MAPE":    round(mape, 6),
            "N":       n,
            "source_folder": name,
        })
    return rows


def iter_folders(include_seeds: bool):
    """Yield all model folders we should process."""
    yield from MODELS_DIR.iterdir()
    if include_seeds:
        for sub in ["SEED123", "seed777"]:
            d = MODELS_DIR / sub
            if d.exists():
                yield from d.iterdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-seed-folders", action="store_true",
                    help="Also process models/SEED123 and models/seed777")
    args = ap.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"📂 Scanning: {MODELS_DIR}")
    if args.include_seed_folders:
        print(f"   (including SEED123/, seed777/)")

    all_rows = []
    for folder in iter_folders(args.include_seed_folders):
        if not folder.is_dir():
            continue
        rows = process_folder(folder)
        if rows:
            print(f"  ✓ {folder.name}  →  {len(rows)} regime rows")
            all_rows.extend(rows)

    if not all_rows:
        print("❌ No predictions found. Make sure training has produced "
              "predictions_mph.npy + targets_mph.npy in models/*/ folders.")
        return 1

    df = pd.DataFrame(all_rows).sort_values(
        ["dataset", "model", "config", "Q", "regime"]
    )
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n💾 Saved {len(df)} rows to: {OUTPUT_PATH}")
    print(f"\n{'='*70}")
    print("REGIME BREAKDOWN SUMMARY (averaged across model+config, per Q)")
    print(f"{'='*70}")
    pivot = df.pivot_table(
        index=["dataset", "Q", "regime"],
        values="MAE",
        aggfunc="mean",
    ).round(3)
    print(pivot.to_string())
    print()
    print("Use this CSV to add a new Table to Section VI of the manuscript that "
          "addresses EIC-3 (per-regime evolution of speed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
