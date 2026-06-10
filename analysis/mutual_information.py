#!/usr/bin/env python3
"""
PHASE 3B — Mutual Information + Granger Causality Analysis
===========================================================
IEEE-TITS Revision T-ITS-26-01-0232 | Addresses: R2-6, R3-2

Computes:
  1. Mutual Information (MI) between acceleration at time t and speed at t+k
     for k = 1, 2, 3, 6, 12.  Averaged across all sensors (training split).
  2. Granger Causality: tests whether past acceleration (lags 1-12) Granger-
     causes future speed. Uses statsmodels. Sensor 61 for METR-LA,
     sensor 111 for PEMS-BAY.

Data: training split only (first 70% of timestamps, matches paper).
Data dirs:
  METR-LA : data/metr-la-v2/     (SG-filtered, v2 train-only normalization)
  PEMS-BAY: data/pems-bay/       (SG-filtered)

Outputs:
  results/mutual_information_results.csv
  results/granger_causality_results.csv

Required packages: scikit-learn, statsmodels
    pip install scikit-learn statsmodels

Run from STGNN root:
    python revision/phase3B_mutual_information.py
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent   # STGNN/

try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    raise ImportError("statsmodels required: pip install statsmodels")


SEQ_LEN    = 12
TRAIN_RATIO = 0.7
FUTURE_STEPS = [1, 2, 3, 6, 12]
GRANGER_LAGS = list(range(1, 13))
GRANGER_SENSOR = {"metr-la": 61, "pems-bay": 111}

DATASETS = {
    "metr-la": {
        "data_dir": ROOT / "data/metr-la-v2",
        "speed_file": "scaled_speed.npy",
        "accel_file": "scaled_acceleration.npy",
    },
    "pems-bay": {
        "data_dir": ROOT / "data/pems-bay",
        "speed_file": "scaled_speed.npy",
        "accel_file": "scaled_acceleration.npy",
    },
}

OUTPUT_DIR = ROOT / "results"


def load_data(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    data_dir   = cfg["data_dir"]
    speed_path = data_dir / cfg["speed_file"]
    accel_path = data_dir / cfg["accel_file"]

    if not speed_path.exists():
        raise FileNotFoundError(f"Speed data not found: {speed_path}")
    if not accel_path.exists():
        raise FileNotFoundError(f"Acceleration data not found: {accel_path}")

    speed = np.load(speed_path).astype(np.float64)  # (T, N)
    accel = np.load(accel_path).astype(np.float64)  # (T, N)
    return speed, accel


def get_training_split(data: np.ndarray, train_ratio=TRAIN_RATIO) -> np.ndarray:
    num_samples = data.shape[0] - SEQ_LEN - 1
    train_size  = int(train_ratio * num_samples)
    return data[:train_size + SEQ_LEN]


def compute_mi_per_step(accel_train: np.ndarray, speed_train: np.ndarray,
                         future_steps: list) -> pd.DataFrame:
    """
    MI between accel[t] (all sensors averaged) and speed[t+k].
    Uses sklearn mutual_info_regression.
    Returns DataFrame with columns: future_step, mi_mean, mi_std.
    """
    T, N = accel_train.shape
    rows = []

    print(f"   Computing MI for {N} sensors × {len(future_steps)} future steps...")

    for k in future_steps:
        # Build (X, y) pairs: X = accel at t, y = speed at t+k
        # Valid indices: t can range from 0 to T-k-1
        max_t = T - k - 1
        if max_t < 50:
            print(f"   ⚠  Not enough samples for step k={k}")
            continue

        X = accel_train[:max_t]   # (max_t, N)
        Y = speed_train[k:max_t + k]  # (max_t, N)

        mi_per_sensor = []
        for s in range(N):
            x_s = X[:, s].reshape(-1, 1)
            y_s = Y[:, s]
            try:
                mi = mutual_info_regression(x_s, y_s, random_state=42)[0]
                mi_per_sensor.append(mi)
            except Exception:
                pass

        mi_mean = float(np.mean(mi_per_sensor)) if mi_per_sensor else float("nan")
        mi_std  = float(np.std(mi_per_sensor))  if mi_per_sensor else float("nan")

        rows.append({
            "future_step": k,
            "mi_mean": round(mi_mean, 6),
            "mi_std":  round(mi_std, 6),
            "n_sensors": len(mi_per_sensor),
        })
        print(f"   k={k:2d}: MI = {mi_mean:.4f} ± {mi_std:.4f}")

    return pd.DataFrame(rows)


def compute_granger(accel_train: np.ndarray, speed_train: np.ndarray,
                    sensor_idx: int, max_lag: int = 12) -> pd.DataFrame:
    """
    Granger causality: does past acceleration (lags 1..max_lag) Granger-cause future speed?
    Uses statsmodels grangercausalitytests on a single sensor time series.
    Returns DataFrame with columns: lag, F_stat, p_value.
    """
    accel_s = accel_train[:, sensor_idx]
    speed_s = speed_train[:, sensor_idx]

    data_matrix = np.column_stack([speed_s, accel_s])

    print(f"   Running Granger causality on sensor {sensor_idx}...")
    try:
        results = grangercausalitytests(
            data_matrix, maxlag=max_lag, verbose=False
        )
    except Exception as e:
        print(f"   ❌ Granger test failed: {e}")
        return pd.DataFrame()

    rows = []
    for lag, res in results.items():
        f_test = res[0]["ssr_ftest"]
        f_stat = f_test[0]
        p_val  = f_test[1]
        rows.append({
            "lag": lag,
            "F_stat":  round(float(f_stat), 4),
            "p_value": round(float(p_val),  6),
            "significant_at_05": p_val < 0.05,
        })

    df = pd.DataFrame(rows)
    print(f"   Significant lags (p<0.05): {df[df['significant_at_05']]['lag'].tolist()}")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PHASE 3B — Mutual Information + Granger Causality")
    print(f"{'='*70}\n")

    mi_all_rows      = []
    granger_all_rows = []

    for dataset_name, cfg in DATASETS.items():
        print(f"📂 Dataset: {dataset_name}")
        try:
            speed, accel = load_data(cfg)
        except FileNotFoundError as e:
            print(f"  ❌ {e}\n")
            continue

        print(f"   Speed: {speed.shape}  Accel: {accel.shape}")

        speed_train = get_training_split(speed)
        accel_train = get_training_split(accel)
        print(f"   Training split: {speed_train.shape}")

        # ── Mutual Information ──
        print("\n   [MI Analysis]")
        mi_df = compute_mi_per_step(accel_train, speed_train, FUTURE_STEPS)
        mi_df.insert(0, "dataset", dataset_name)
        mi_all_rows.append(mi_df)

        # ── Granger Causality ──
        sensor_idx = GRANGER_SENSOR[dataset_name]
        print(f"\n   [Granger Causality — sensor {sensor_idx}]")
        granger_df = compute_granger(accel_train, speed_train, sensor_idx, max_lag=12)
        if not granger_df.empty:
            granger_df.insert(0, "sensor", sensor_idx)
            granger_df.insert(0, "dataset", dataset_name)
            granger_all_rows.append(granger_df)

        print()

    # Save MI results
    if mi_all_rows:
        mi_final = pd.concat(mi_all_rows, ignore_index=True)
        mi_path  = OUTPUT_DIR / "mutual_information_results.csv"
        mi_final.to_csv(mi_path, index=False)
        print(f"✅ MI results saved → {mi_path}")
        print(f"\n{mi_final.to_string(index=False)}\n")
    else:
        print("⚠  No MI results generated.")

    # Save Granger results
    if granger_all_rows:
        granger_final = pd.concat(granger_all_rows, ignore_index=True)
        granger_path  = OUTPUT_DIR / "granger_causality_results.csv"
        granger_final.to_csv(granger_path, index=False)
        print(f"✅ Granger results saved → {granger_path}")
        print(f"\n{granger_final.to_string(index=False)}\n")
    else:
        print("⚠  No Granger results generated.")

    print(f"\n{'='*70}")
    print("PHASE 3B COMPLETE")
    print("Use results/mutual_information_results.csv for Fig. 10")
    print("and the Granger results in Section VII Discussion.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
