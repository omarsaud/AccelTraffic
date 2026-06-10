#!/usr/bin/env python3
"""
===============================================================================
PHASE 6 — ABLATION SUMMARY AGGREGATOR  (R1.2 / R2.1 / R2.2)
===============================================================================
NO training. Reads existing test_results.json files and builds three clean
comparison tables, one per reviewer ablation:

  speedspeed   (R1.2)  : SpeedSpeed (2 speed channels) vs NoAcc vs Acc_SG
                         → tests kinematics vs mere dimensionality
  unified_norm (R2.1)  : Acc_SG_UnifiedNorm vs Acc_SG
                         → tests necessity of dual-channel normalization
  accel_only   (R2.2)  : AccelOnly vs NoAcc vs Acc_SG
                         → tests that speed is necessary; accel is complementary

Grid: 5 models × 2 datasets × 3 horizons = 30 cells per ablation.

Baselines (NoAcc / Acc_NoSG / Acc_SG) are read from:
    models/{model}_{dataset}_{config}_LSTM_Q{Q}/test_results.json
(STGIN folders follow the same pattern; all STGIN is the STE-off configuration
per the manuscript convention — see audit/08 decision D-8.)

Ablation results are read from:
    results/{ablation}/{...}_Q{Q}/test_results.json

Outputs (results/):
    ablation_speedspeed.csv
    ablation_unifiednorm.csv
    ablation_accelonly.csv

Run:
    python revision/phase6_ablation_summary.py
===============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REV_DIR    = ROOT / "results"

MODELS   = ["agcrn", "dcrnn", "gwnet", "stgin", "staeformer"]
DATASETS = ["metr-la", "pems-bay"]
HORIZONS = [3, 6, 12]


def _read_mae(folder: Path):
    """
    Return denormalized test MAE (mph) from a folder, handling BOTH schemas:
      - test_results.json  → key 'test_mae'           (multi-model + revision pipeline)
      - test_metrics.json  → 'denormalized_metrics'.'mae_mph'  (STGIN pipeline)
    Returns None if neither is found.
    """
    # Schema 1: test_results.json
    p1 = folder / "test_results.json"
    if p1.exists():
        try:
            d = json.load(open(p1))
            v = d.get("test_mae", None)
            if v is not None:
                return float(v)
        except Exception:
            pass
    # Schema 2: test_metrics.json (STGIN baselines)
    p2 = folder / "test_metrics.json"
    if p2.exists():
        try:
            d = json.load(open(p2))
            dm = d.get("denormalized_metrics", {})
            v = dm.get("mae_mph", dm.get("mae", None))
            if v is not None:
                return float(v)
        except Exception:
            pass
    return None


def baseline_mae(model, dataset, config, Q):
    """
    Baseline folders: models/{model}_{dataset}_{config}_LSTM_Q{Q}
    Fall back to a glob if the exact name differs (e.g., missing _LSTM).
    """
    exact = MODELS_DIR / f"{model}_{dataset}_{config}_LSTM_Q{Q}"
    v = _read_mae(exact)
    if v is not None:
        return v
    # fallback: any folder starting with model_dataset_config and ending _Q{Q}
    for d in MODELS_DIR.glob(f"{model}_{dataset}_{config}*_Q{Q}"):
        if d.is_dir():
            v = _read_mae(d)
            if v is not None:
                return v
    return None


def ablation_mae(ablation_subdir, model, dataset, Q):
    """
    Find the ablation result folder for (model, dataset, Q) inside
    results/{ablation_subdir}/ regardless of the exact config token
    (handles both '{model}_{dataset}_<cfg>_Q{Q}' and the STGIN
    '{model}_{dataset}_<cfg>_LSTM_Q{Q}' naming).
    """
    base = REV_DIR / ablation_subdir
    if not base.exists():
        return None
    for d in base.glob(f"{model}_{dataset}_*_Q{Q}"):
        if d.is_dir():
            v = _read_mae(d)
            if v is not None:
                return v
    return None


def pct(better, worse):
    """% improvement of `better` (lower MAE) relative to `worse`."""
    if better is None or worse is None or worse == 0:
        return None
    return round((worse - better) / worse * 100.0, 2)


def build():
    speedspeed_rows, unified_rows, accel_rows = [], [], []

    for model in MODELS:
        for ds in DATASETS:
            for Q in HORIZONS:
                noacc   = baseline_mae(model, ds, "NoAcc", Q)
                accnosg = baseline_mae(model, ds, "Acc_NoSG", Q)
                accsg   = baseline_mae(model, ds, "Acc_SG", Q)

                # --- R1.2 SpeedSpeed ---
                ss = ablation_mae("speedspeed", model, ds, Q)
                speedspeed_rows.append({
                    "model": model, "dataset": ds, "Q": Q,
                    "NoAcc": noacc, "Acc_SG": accsg, "SpeedSpeed": ss,
                    "SpeedSpeed_vs_NoAcc_%": pct(ss, noacc),       # ~0 expected (no help)
                    "AccSG_vs_SpeedSpeed_%": pct(accsg, ss),      # large + expected (kinematics)
                })

                # --- R2.1 UnifiedNorm ---
                un = ablation_mae("unified_norm", model, ds, Q)
                unified_rows.append({
                    "model": model, "dataset": ds, "Q": Q,
                    "Acc_SG_dual": accsg, "Acc_SG_UnifiedNorm": un,
                    "DualNorm_improvement_%": pct(accsg, un),     # + expected (dual wins)
                })

                # --- R2.2 AccelOnly ---
                ao = ablation_mae("accel_only", model, ds, Q)
                accel_rows.append({
                    "model": model, "dataset": ds, "Q": Q,
                    "NoAcc": noacc, "Acc_SG": accsg, "AccelOnly": ao,
                    "AccelOnly_vs_NoAcc_%": pct(ao, noacc),       # negative expected (worse than speed)
                    "AccSG_vs_AccelOnly_%": pct(accsg, ao),       # large + expected
                })

    return (pd.DataFrame(speedspeed_rows),
            pd.DataFrame(unified_rows),
            pd.DataFrame(accel_rows))


def coverage(df, *value_cols):
    """Count non-null cells for the ablation value columns."""
    return {c: int(df[c].notna().sum()) for c in value_cols}


def main():
    REV_DIR.mkdir(parents=True, exist_ok=True)
    ss, un, ao = build()

    ss.to_csv(REV_DIR / "ablation_speedspeed.csv", index=False)
    un.to_csv(REV_DIR / "ablation_unifiednorm.csv", index=False)
    ao.to_csv(REV_DIR / "ablation_accelonly.csv", index=False)

    print("=" * 78)
    print("ABLATION SUMMARY (R1.2 / R2.1 / R2.2) — no retraining, read from disk")
    print("=" * 78)

    print("\n--- R1.2 SpeedSpeed (kinematics vs dimensionality) ---")
    print(f"   coverage: {coverage(ss, 'SpeedSpeed')}")
    print(ss.to_string(index=False))

    print("\n--- R2.1 UnifiedNorm (necessity of dual-channel normalization) ---")
    print(f"   coverage: {coverage(un, 'Acc_SG_UnifiedNorm')}")
    print(un.to_string(index=False))

    print("\n--- R2.2 AccelOnly (speed is necessary; accel complementary) ---")
    print(f"   coverage: {coverage(ao, 'AccelOnly')}")
    print(ao.to_string(index=False))

    # Headline checks for the response letter
    print("\n" + "=" * 78)
    print("HEADLINE CHECKS")
    print("=" * 78)
    ss_valid = ss.dropna(subset=["Acc_SG", "SpeedSpeed"])
    n_accsg_beats_ss = int((ss_valid["Acc_SG"] < ss_valid["SpeedSpeed"]).sum())
    print(f"R1.2: Acc_SG beats SpeedSpeed in {n_accsg_beats_ss}/{len(ss_valid)} cells")
    un_valid = un.dropna(subset=["Acc_SG_dual", "Acc_SG_UnifiedNorm"])
    n_dual = int((un_valid["Acc_SG_dual"] < un_valid["Acc_SG_UnifiedNorm"]).sum())
    print(f"R2.1: dual-norm beats unified-norm in {n_dual}/{len(un_valid)} cells")
    ao_valid = ao.dropna(subset=["Acc_SG", "AccelOnly"])
    n_ao = int((ao_valid["Acc_SG"] < ao_valid["AccelOnly"]).sum())
    print(f"R2.2: Acc_SG beats AccelOnly in {n_ao}/{len(ao_valid)} cells")
    print(f"\n💾 Saved 3 CSVs to {REV_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
