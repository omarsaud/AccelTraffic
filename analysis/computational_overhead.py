#!/usr/bin/env python3
"""
PHASE 3C — Computational Overhead Measurement
==============================================
IEEE-TITS Revision T-ITS-26-01-0232 | Addresses: R2-4

Measures:
  1. Time to compute finite-difference acceleration over full METR-LA array
  2. Time to apply causal SG filter (W=13, p=1) over full acceleration array
  3. Per-sensor-per-step cost breakdown
  4. AGCRN inference overhead: NoAcc vs Acc_SG (100 forward passes, batch=64)

All timing uses time.perf_counter() for high resolution.
GPU inference timing uses CUDA events if GPU available, otherwise CPU timing.

Output: results/computational_overhead.csv

Run from STGNN root:
    python revision/phase3C_computational_overhead.py
"""

from __future__ import annotations
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent   # STGNN/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

OUTPUT_DIR      = ROOT / "results"

METR_LA_DIR     = ROOT / "data/metr-la-v2"
NOACC_MODEL_DIR = ROOT / "models/agcrn_metr-la_NoAcc_LSTM_Q3"
ACCSG_MODEL_DIR = ROOT / "models/agcrn_metr-la_Acc_SG_LSTM_Q3"

BATCH_SIZE  = 64
N_FORWARD   = 100
SEQ_LEN     = 12
HORIZON     = 3
SG_WINDOW   = 13
SG_POLY     = 1
REPEATS     = 5


def causal_sg_filter_array(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    Vectorized causal SG filter using scipy.signal.lfilter.
    Computes SG coefficients once (pos=window-1 = evaluate at newest sample),
    then applies them to ALL columns in one C-level call via lfilter.
    Same math as a sample-by-sample loop — but ~10,000x faster.
    Startup transient: first (window-1) samples use implicit zero-padding,
    which is acceptable for a timing measurement.
    """
    from scipy.signal import savgol_coeffs, lfilter
    # pos=window-1: evaluate polynomial at the newest (last) point → causal
    coeffs = savgol_coeffs(window, polyorder, pos=window - 1, use="conv")
    # lfilter(b, [1], x, axis=0): y[n] = sum_k b[k]*x[n-k]  (pure FIR, causal)
    return lfilter(coeffs, [1.0], arr.astype(np.float64), axis=0)


def time_operation(fn, repeats=REPEATS) -> dict:
    """Run fn() `repeats` times and return timing stats (ms)."""
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms":  float(np.std(times_ms)),
        "min_ms":  float(np.min(times_ms)),
    }


def measure_preprocessing(speed: np.ndarray) -> list:
    T, N = speed.shape
    print(f"   Data shape: {T} timesteps × {N} sensors")
    rows = []

    # --- Finite difference acceleration ---
    print("   [1] Finite-difference acceleration...")

    def compute_accel():
        accel = np.diff(speed, axis=0)
        accel = np.vstack([accel[:1], accel])
        return accel

    timing = time_operation(compute_accel)
    per_sensor_per_step = timing["mean_ms"] / (T * N)
    print(f"      Mean: {timing['mean_ms']:.2f} ms ± {timing['std_ms']:.2f} ms")
    print(f"      Per-sensor-per-step: {per_sensor_per_step*1000:.4f} µs")
    rows.append({
        "operation": "Finite-difference acceleration",
        "total_ms_mean": round(timing["mean_ms"], 3),
        "total_ms_std":  round(timing["std_ms"], 3),
        "per_sensor_per_step_us": round(per_sensor_per_step * 1000, 6),
        "array_shape": f"{T}×{N}",
    })

    # --- Causal SG filter ---
    accel_raw = compute_accel()

    if savgol_filter is not None:
        print(f"   [2] Causal SG filter (W={SG_WINDOW}, p={SG_POLY}) — may take a moment...")

        def apply_sg():
            return causal_sg_filter_array(accel_raw, SG_WINDOW, SG_POLY)

        timing_sg = time_operation(apply_sg, repeats=2)
        per_sg = timing_sg["mean_ms"] / (T * N)
        print(f"      Mean: {timing_sg['mean_ms']:.1f} ms ± {timing_sg['std_ms']:.1f} ms")
        print(f"      Per-sensor-per-step: {per_sg*1000:.4f} µs")
        rows.append({
            "operation": f"Causal SG filter (W={SG_WINDOW}, p={SG_POLY})",
            "total_ms_mean": round(timing_sg["mean_ms"], 3),
            "total_ms_std":  round(timing_sg["std_ms"], 3),
            "per_sensor_per_step_us": round(per_sg * 1000, 6),
            "array_shape": f"{T}×{N}",
        })
    else:
        print("   [2] Causal SG filter — scipy not installed, skipping.")

    return rows


def measure_inference(model_dir: Path, label: str, input_dim: int,
                      num_nodes: int) -> Optional[dict]:
    """
    Load AGCRN best_model.pt and time 100 forward passes at batch_size=64.
    Returns dict with timing info.
    """
    model_pt   = model_dir / "best_model.pt"
    config_pt  = model_dir / "model_config.json"
    norm_pt    = model_dir / "normalization_params.json"

    if not model_pt.exists():
        print(f"   ⚠  Model not found: {model_pt}")
        return None

    try:
        import torch
        from codes.model_factory import create_model
    except ImportError as e:
        print(f"   ⚠  Cannot import torch/model_factory: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model_cfg = {}
    if config_pt.exists():
        with open(config_pt) as f:
            model_cfg = json.load(f)

    arch = model_cfg.get("architecture", {})
    hidden_dim = arch.get("hidden_dim", 64)
    dropout    = arch.get("dropout", 0.3)

    model = create_model(
        model_name="agcrn",
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=hidden_dim,
        historical_window=SEQ_LEN,
        prediction_horizon=HORIZON,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.to(device).eval()

    adj_path = METR_LA_DIR / "adj_mx.pkl"
    if adj_path.exists():
        import pickle
        with open(adj_path, "rb") as f:
            _, _, adj = pickle.load(f, encoding="latin1")
        adj_t = torch.FloatTensor(adj).to(device)
    else:
        adj_t = torch.eye(num_nodes).to(device)

    dummy_x = torch.randn(BATCH_SIZE, num_nodes, SEQ_LEN, input_dim).to(device)

    with torch.no_grad():
        _ = model(dummy_x, adj_t)

    times_ms = []
    with torch.no_grad():
        for _ in range(N_FORWARD):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_x, adj_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    mean_ms = float(np.mean(times_ms))
    std_ms  = float(np.std(times_ms))
    per_sample_ms = mean_ms / BATCH_SIZE

    print(f"   {label}: {mean_ms:.2f} ± {std_ms:.2f} ms per batch | {per_sample_ms:.3f} ms per sample")

    return {
        "operation": f"AGCRN inference ({label})",
        "total_ms_mean": round(mean_ms, 3),
        "total_ms_std":  round(std_ms, 3),
        "per_sensor_per_step_us": round(per_sample_ms * 1000, 3),
        "array_shape": f"batch={BATCH_SIZE}, nodes={num_nodes}, seq={SEQ_LEN}, dim={input_dim}",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PHASE 3C — Computational Overhead Measurement")
    print(f"{'='*70}\n")

    rows = []

    # --- Load raw speed for preprocessing timing ---
    speed_path = METR_LA_DIR / "scaled_speed.npy"
    if not speed_path.exists():
        print(f"❌ Speed data not found: {speed_path}")
        return

    speed = np.load(speed_path).astype(np.float64)
    T, N  = speed.shape
    print(f"📂 Loaded METR-LA: {T} × {N}")
    print(f"\n[PREPROCESSING TIMING] (repeats={REPEATS})")

    preproc_rows = measure_preprocessing(speed)
    rows.extend(preproc_rows)

    # --- Inference timing ---
    print(f"\n[INFERENCE TIMING] ({N_FORWARD} forward passes, batch={BATCH_SIZE})")

    result_noacc = measure_inference(NOACC_MODEL_DIR, "NoAcc (1ch)", input_dim=1, num_nodes=N)
    if result_noacc:
        rows.append(result_noacc)

    result_accsg = measure_inference(ACCSG_MODEL_DIR, "Acc_SG (2ch)", input_dim=2, num_nodes=N)
    if result_accsg:
        rows.append(result_accsg)

    # Overhead ratio
    if result_noacc and result_accsg:
        overhead_pct = (result_accsg["total_ms_mean"] - result_noacc["total_ms_mean"]) \
                       / result_noacc["total_ms_mean"] * 100
        print(f"\n   Inference overhead (Acc_SG vs NoAcc): {overhead_pct:+.1f}%")
        rows.append({
            "operation": "Inference overhead (Acc_SG vs NoAcc)",
            "total_ms_mean": round(result_accsg["total_ms_mean"] - result_noacc["total_ms_mean"], 4),
            "total_ms_std": float("nan"),
            "per_sensor_per_step_us": round(overhead_pct, 2),
            "array_shape": "overhead_percent",
        })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "computational_overhead.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved → {csv_path}")
    print(f"\n{df[['operation', 'total_ms_mean', 'total_ms_std']].to_string(index=False)}")

    print(f"\n{'='*70}")
    print("PHASE 3C COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
