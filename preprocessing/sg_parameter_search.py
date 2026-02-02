"""
Causal Savitzky-Golay Parameter Search for Acceleration Preprocessing
======================================================================

This script performs a grid search to find optimal SG filter parameters
for acceleration preprocessing in traffic speed prediction.

Methodology:
1. Use TRAINING data only (70% split) to prevent data leakage
2. Apply CAUSAL filtering (only past data, no future)
3. Evaluate based on PREDICTIVE CORRELATION with future speed
4. Secondary criterion: smoothness (variance reduction)

Optimal parameters found: W=13, P=1

Usage:
    python sg_parameter_search.py --dataset metr-la
    python sg_parameter_search.py --dataset pems-bay

Reference:
    Aba Hussen et al., "Acceleration-Driven Deep Learning for Traffic Speed
    Prediction with Causal Filtering and Dual-Channel Normalization"
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def causal_sg_filter_1d(data, window, polyorder):
    """
    Apply CAUSAL Savitzky-Golay-style polynomial filter.
    
    At time t, uses ONLY data from [t-window+1, ..., t].
    No future data = No data leakage.
    
    Args:
        data: 1D numpy array (timesteps,)
        window: Window size (must be > polyorder)
        polyorder: Polynomial order for fitting
    
    Returns:
        Filtered 1D array (same length as input)
    """
    result = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < window - 1:
            # Not enough past data, use available
            window_data = data[:i+1]
            if len(window_data) >= polyorder + 1:
                x = np.arange(len(window_data))
                coeffs = np.polyfit(x, window_data, polyorder)
                result[i] = np.polyval(coeffs, len(window_data)-1)
            else:
                result[i] = data[i]
        else:
            # Full window available
            window_data = data[i-window+1:i+1]
            x = np.arange(window)
            coeffs = np.polyfit(x, window_data, polyorder)
            result[i] = np.polyval(coeffs, window-1)
    
    return result


def run_grid_search(speed, accel, horizon=12, window_sizes=None, poly_orders=None):
    """
    Grid search for optimal SG filter parameters.
    
    Criterion: Maximize correlation of filtered_accel[t] with speed[t+horizon]
    
    Args:
        speed: Speed array (T, N) - normalized
        accel: Acceleration array (T, N) - normalized (unfiltered)
        horizon: Prediction horizon (default: 12 = 60 minutes)
        window_sizes: List of window sizes to test
        poly_orders: List of polynomial orders to test
    
    Returns:
        DataFrame with results, optimal parameters dict
    """
    if window_sizes is None:
        window_sizes = [7, 9, 11, 13, 15]
    if poly_orders is None:
        poly_orders = [1, 2, 3]
    
    T_total, N = speed.shape
    
    # Use training set only (70%)
    train_size = int(T_total * 0.7)
    speed_train = speed[:train_size]
    accel_train = accel[:train_size]
    
    print(f"Total: T={T_total}, N={N}")
    print(f"Training: T={train_size} (70%)")
    print(f"Horizon: {horizon} steps ({horizon * 5} minutes)")
    print(f"Window sizes: {window_sizes}")
    print(f"Polynomial orders: {poly_orders}")
    
    # Prepare aligned data for correlation
    speed_future = speed_train[horizon:]
    accel_current = accel_train[:-horizon]
    
    # Sample sensors for faster search (use all if < 50)
    num_sensors = min(50, N)
    sensor_idx = np.linspace(0, N-1, num_sensors, dtype=int)
    
    print(f"Using {num_sensors} sensors for grid search")
    
    results = []
    
    for window in tqdm(window_sizes, desc="Window sizes"):
        for poly in poly_orders:
            # Skip invalid combinations
            if poly >= window:
                continue
            
            try:
                # Apply causal filter to sampled sensors
                corr_list = []
                raw_var_list = []
                filt_var_list = []
                
                for s in sensor_idx:
                    # Filter this sensor's acceleration
                    accel_filtered = causal_sg_filter_1d(
                        accel_current[:, s], window, poly
                    )
                    
                    # Compute correlation with future speed
                    if speed_future[:, s].std() > 0 and accel_filtered.std() > 0:
                        corr, _ = spearmanr(accel_filtered, speed_future[:, s])
                        if not np.isnan(corr):
                            corr_list.append(abs(corr))
                    
                    # Variance for smoothness
                    raw_var_list.append(np.var(accel_current[:, s]))
                    filt_var_list.append(np.var(accel_filtered))
                
                mean_corr = np.mean(corr_list) if corr_list else 0
                
                # Smoothness: variance reduction
                raw_var = np.mean(raw_var_list)
                filt_var = np.mean(filt_var_list)
                smoothness = 1 - (filt_var / raw_var) if raw_var > 0 else 0
                
                results.append({
                    'window': window,
                    'polyorder': poly,
                    'correlation': mean_corr,
                    'smoothness': smoothness,
                    'window_minutes': window * 5,
                })
                
            except Exception as e:
                print(f"Error with W={window}, P={poly}: {e}")
                continue
    
    # Create DataFrame and sort by correlation (primary), smoothness (secondary)
    df = pd.DataFrame(results)
    df = df.sort_values(['correlation', 'smoothness'], ascending=[False, False])
    df = df.reset_index(drop=True)
    
    # Get optimal
    if len(df) > 0:
        optimal = df.iloc[0]
        optimal_params = {
            'window': int(optimal['window']),
            'polyorder': int(optimal['polyorder']),
            'window_minutes': int(optimal['window_minutes']),
            'correlation': float(optimal['correlation']),
            'smoothness': float(optimal['smoothness']),
            'horizon': horizon,
            'filtering_type': 'causal'
        }
    else:
        optimal_params = None
    
    return df, optimal_params


def plot_heatmaps(df, dataset_name, output_dir):
    """Create heatmap visualizations of grid search results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'SG Parameter Search: {dataset_name}', fontsize=14, fontweight='bold')
    
    # Correlation heatmap
    pivot_corr = df.pivot(index='polyorder', columns='window', values='correlation')
    sns.heatmap(pivot_corr, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[0],
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Predictive Correlation (higher = better)')
    axes[0].set_xlabel('Window Size (timesteps)')
    axes[0].set_ylabel('Polynomial Order')
    
    # Smoothness heatmap
    pivot_smooth = df.pivot(index='polyorder', columns='window', values='smoothness')
    sns.heatmap(pivot_smooth, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
                cbar_kws={'label': 'Smoothness'})
    axes[1].set_title('Smoothness (variance reduction)')
    axes[1].set_xlabel('Window Size (timesteps)')
    axes[1].set_ylabel('Polynomial Order')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'sg_parameter_search_{dataset_name.lower().replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap: {output_dir / f'sg_parameter_search_{dataset_name.lower().replace('-', '_')}.png'}")


def main():
    parser = argparse.ArgumentParser(description='SG Parameter Search for Acceleration')
    parser.add_argument('--dataset', type=str, default='metr-la',
                        choices=['metr-la', 'pems-bay'],
                        help='Dataset name')
    parser.add_argument('--horizon', type=int, default=12,
                        help='Prediction horizon in timesteps')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: ../data)')
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).resolve().parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir.parent / 'data'
    
    output_dir = script_dir.parent / 'figures'
    
    print("=" * 70)
    print("CAUSAL SG PARAMETER SEARCH FOR ACCELERATION PREPROCESSING")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Horizon: {args.horizon} steps ({args.horizon * 5} minutes)")
    print(f"Data directory: {data_dir}")
    
    # Load data
    # Look for unfiltered data first, then regular
    dataset_dirs = [
        data_dir / f"{args.dataset}-unfiltered",
        data_dir / args.dataset,
    ]
    
    data_path = None
    for d in dataset_dirs:
        if d.exists():
            data_path = d
            break
    
    if data_path is None:
        print(f"\nERROR: Dataset not found in {data_dir}")
        print("Please ensure data is downloaded and preprocessed.")
        return
    
    print(f"Using data from: {data_path}")
    
    # Load normalized speed and acceleration
    speed_file = data_path / 'scaled_speed.npy'
    accel_file = data_path / 'scaled_acceleration.npy'
    
    if not speed_file.exists() or not accel_file.exists():
        print(f"\nERROR: Missing files:")
        print(f"  - {speed_file}: {'EXISTS' if speed_file.exists() else 'MISSING'}")
        print(f"  - {accel_file}: {'EXISTS' if accel_file.exists() else 'MISSING'}")
        print("\nPlease run generate_acceleration.py first.")
        return
    
    speed = np.load(speed_file)
    accel = np.load(accel_file)
    
    print(f"\nLoaded data:")
    print(f"  Speed: {speed.shape}")
    print(f"  Acceleration: {accel.shape}")
    
    # Run grid search
    print("\n" + "=" * 70)
    print("RUNNING GRID SEARCH")
    print("=" * 70)
    
    df, optimal = run_grid_search(speed, accel, horizon=args.horizon)
    
    # Display results
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 70)
    print(df.head(10).to_string(index=False))
    
    if optimal:
        print("\n" + "=" * 70)
        print("OPTIMAL PARAMETERS")
        print("=" * 70)
        print(f"  Window size: {optimal['window']} timesteps ({optimal['window_minutes']} minutes)")
        print(f"  Polynomial order: {optimal['polyorder']}")
        print(f"  Correlation: {optimal['correlation']:.4f}")
        print(f"  Smoothness: {optimal['smoothness']:.4f}")
        print("=" * 70)
        
        # Save results
        df.to_csv(data_path / 'sg_parameter_search_results.csv', index=False)
        print(f"\nSaved: {data_path / 'sg_parameter_search_results.csv'}")
        
        with open(data_path / 'optimal_sg_parameters.json', 'w') as f:
            json.dump(optimal, f, indent=2)
        print(f"Saved: {data_path / 'optimal_sg_parameters.json'}")
        
        # Create visualizations
        plot_heatmaps(df, args.dataset.upper(), output_dir)
    
    print("\n✅ Parameter search complete!")


if __name__ == '__main__':
    main()
