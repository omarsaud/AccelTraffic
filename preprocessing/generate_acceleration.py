"""
Generate acceleration channels from speed data using causal Savitzky-Golay filtering.

This script:
1. Loads raw speed data from METR-LA or PEMS-BAY
2. Computes acceleration as the time derivative of speed
3. Applies strictly causal SG filtering (W=13, p=1)
4. Computes dual-channel normalization statistics
5. Saves normalized speed and acceleration channels

Usage:
    python generate_acceleration.py --dataset metr-la
    python generate_acceleration.py --dataset pems-bay
"""

import argparse
import numpy as np
import h5py
import json
import pandas as pd
from pathlib import Path
# Note: We implement TRUE causal SG filter using polyfit, not scipy.savgol_filter
# scipy.savgol_filter is centered and would leak future data even with padding


def load_speed_data(data_dir, dataset):
    """Load raw speed data from H5 file."""
    h5_path = data_dir / dataset / f"{dataset}.h5"
    
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {h5_path}\n"
            f"Please download from: https://github.com/liyaguang/DCRNN"
        )
    
    with h5py.File(h5_path, 'r') as f:
        # DCRNN format: 'df' dataset with 'block0_values'
        if 'df' in f:
            speed = f['df']['block0_values'][:]
        else:
            # Alternative format
            speed = f['speed'][:]
    
    print(f"Loaded speed data: {speed.shape}")
    return speed


def compute_acceleration(speed, dt=5*60):
    """
    Compute acceleration from speed using finite differences.
    
    Args:
        speed: Speed array (timesteps, sensors)
        dt: Time interval in seconds (default: 5 minutes = 300s)
    
    Returns:
        Acceleration array (same shape as speed)
    """
    # Convert mph to m/s for physical units
    speed_ms = speed * 0.44704  # mph to m/s
    
    # Finite difference: a[t] = (v[t] - v[t-1]) / dt
    accel = np.zeros_like(speed_ms)
    accel[1:] = (speed_ms[1:] - speed_ms[:-1]) / dt
    accel[0] = accel[1]  # Pad first value
    
    return accel


def causal_sg_filter_1d(data, window, poly=1):
    """
    Apply TRUE causal Savitzky-Golay filter to 1D signal.
    
    CRITICAL: This implementation uses polynomial fitting on a BACKWARD-LOOKING
    window only. For window=13, at time t, we use samples [t-12, ..., t-1, t].
    NO FUTURE DATA IS EVER USED.
    
    This is fundamentally different from scipy.savgol_filter which uses a
    CENTERED window and would leak future data even with padding tricks.
    """
    result = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < window - 1:
            # Not enough past samples - use available data
            w_data = data[:i+1]
            if len(w_data) >= poly + 1:
                x = np.arange(len(w_data))
                coeffs = np.polyfit(x, w_data, poly)
                result[i] = np.polyval(coeffs, len(w_data) - 1)
            else:
                result[i] = data[i]
        else:
            # Full window available: [i-window+1, ..., i] = window samples
            w_data = data[i - window + 1 : i + 1]
            x = np.arange(window)
            coeffs = np.polyfit(x, w_data, poly)
            result[i] = np.polyval(coeffs, window - 1)
    
    return result


def causal_sg_filter(signal, window_length=13, polyorder=1):
    """
    Apply TRUE causal Savitzky-Golay filter using BACKWARD-LOOKING window.
    
    CRITICAL: This implementation ensures TRUE causality by fitting polynomials
    to backward-looking windows only. For window_length=13, the filter at time t
    uses samples [t-12, t-11, ..., t-1, t] - NO FUTURE INFORMATION.
    
    WARNING: Do NOT use scipy.signal.savgol_filter for causal filtering!
    scipy.savgol_filter uses a CENTERED window which leaks future data
    even when combined with padding tricks.
    
    Args:
        signal: 1D or 2D array (timesteps,) or (timesteps, sensors)
        window_length: Filter window size (e.g., 13)
        polyorder: Polynomial order for fitting (e.g., 1 for linear)
    
    Returns:
        Filtered signal (same shape as input) using ONLY past information
    """
    if signal.ndim == 1:
        return causal_sg_filter_1d(signal, window_length, polyorder)
    else:
        # Apply to each sensor independently
        T, N = signal.shape
        filtered = np.zeros_like(signal)
        for i in range(N):
            filtered[:, i] = causal_sg_filter_1d(signal[:, i], window_length, polyorder)
        return filtered


def compute_normalization_stats(data, train_ratio=0.7, per_sensor=True):
    """
    Compute mean and std from TRAINING SET ONLY (proper ML methodology).
    
    This matches the V2 preprocessing protocol: statistics computed from
    the first 70% (training portion) are applied to normalize all splits.
    
    Args:
        data: Array (timesteps, sensors)
        train_ratio: Fraction of data for training (default: 0.7)
        per_sensor: If True, compute per-sensor stats (recommended).
                   If False, compute global stats across all sensors.
    
    Returns:
        (mean, std) computed from TRAINING data only
        - If per_sensor=True: returns arrays of shape (sensors,)
        - If per_sensor=False: returns scalar values
    
    NOTE: This implements train-only normalization (V2 protocol) which
    prevents any data leakage from validation/test sets. This is the
    standard ML approach used in the V2 experiments.
    """
    # Split data: compute stats from TRAIN only
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    
    if per_sensor:
        # Per-sensor normalization (recommended - matches manuscript)
        mean = np.mean(train_data, axis=0)  # Shape: (N_sensors,)
        std = np.std(train_data, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
    else:
        # Global normalization (alternative - simpler but less flexible)
        mean = float(np.mean(train_data))
        std = float(np.std(train_data))
    
    return mean, std


def normalize(data, mean, std):
    """Z-score normalization."""
    return (data - mean) / std


def main(args):
    data_dir = Path(args.data_dir)
    dataset = args.dataset
    output_dir = data_dir / dataset
    
    print(f"Processing {dataset}...")
    
    # Load speed data
    speed = load_speed_data(data_dir, dataset)
    
    # Compute acceleration
    print("Computing acceleration from speed...")
    accel_raw = compute_acceleration(speed)
    
    # Apply causal SG filter
    print(f"Applying causal SG filter (W={args.window}, p={args.polyorder})...")
    accel_filtered = causal_sg_filter(accel_raw, args.window, args.polyorder)
    
    # Compute variance reduction
    var_raw = np.var(accel_raw)
    var_filtered = np.var(accel_filtered)
    var_reduction = (1 - var_filtered / var_raw) * 100
    print(f"Variance reduction: {var_reduction:.1f}%")
    
    # Compute normalization statistics (training set only)
    print("Computing normalization statistics...")
    mu_speed, sigma_speed = compute_normalization_stats(speed)
    # ★ CORRECTED: Use FILTERED acceleration stats for filtered data
    # This ensures normalized values have std ≈ 1.0 (proper signal strength)
    # The SG benefit is in SMOOTHER PATTERNS, not smaller magnitude!
    mu_accel, sigma_accel = compute_normalization_stats(accel_filtered)
    
    print(f"Speed: μ={mu_speed:.2f} mph, σ={sigma_speed:.2f} mph")
    print(f"Accel: μ={mu_accel:.6f} m/s², σ={sigma_accel:.4f} m/s²")
    
    # Normalize
    speed_norm = normalize(speed, mu_speed, sigma_speed)
    accel_norm = normalize(accel_filtered, mu_accel, sigma_accel)
    
    # Save normalized data
    print("Saving normalized data...")
    np.save(output_dir / 'scaled_speed.npy', speed_norm.astype(np.float32))
    np.save(output_dir / 'scaled_acceleration.npy', accel_norm.astype(np.float32))

    # Save timestamps (if available) for STE correctness
    # NOTE: When training from scaled_speed.npy, the DataFrame index is lost unless we persist it.
    try:
        h5_path = data_dir / dataset / f"{dataset}.h5"
        if h5_path.exists():
            df = pd.read_hdf(h5_path)
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) == speed.shape[0]:
                ts_path = output_dir / 'timestamps.npy'
                np.save(ts_path, df.index.values.astype('datetime64[ns]'))
                print(f"✅ Saved timestamps for STE: {ts_path}")
            else:
                print("⚠️  Could not extract a valid DatetimeIndex from the .h5 file; timestamps.npy not saved")
        else:
            print("⚠️  Source .h5 not found; timestamps.npy not saved")
    except Exception as e:
        print(f"⚠️  Failed to save timestamps.npy: {e}")
    
    # Save normalization parameters
    params = {
        'speed_mean': mu_speed,
        'speed_std': sigma_speed,
        'accel_mean': mu_accel,
        'accel_std': sigma_accel,
        'sg_window': args.window,
        'sg_polyorder': args.polyorder,
        'variance_reduction_pct': var_reduction
    }
    
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Done! Files saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate acceleration channels')
    parser.add_argument('--dataset', type=str, default='metr-la',
                        choices=['metr-la', 'pems-bay'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--window', type=int, default=13,
                        help='SG filter window length')
    parser.add_argument('--polyorder', type=int, default=1,
                        help='SG filter polynomial order')
    
    args = parser.parse_args()
    main(args)
