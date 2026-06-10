"""
07_diebold_mariano_test.py
==========================
Diebold-Mariano (DM) test for pairwise forecast comparison.
Tests whether NoAcc vs Acc_SG MAE differences are statistically significant
accounting for autocorrelation in time-series forecast errors.

Output: results/07_dm_test_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# Diebold-Mariano Test Implementation
# ============================================================

def diebold_mariano_test(e1, e2, horizon=1, power=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    Args:
        e1: forecast errors from model 1 (T,)
        e2: forecast errors from model 2 (T,)
        horizon: forecast horizon (for HAC bandwidth)
        power: 1 for MAE, 2 for MSE
    
    Returns:
        dm_stat: DM test statistic
        p_value: two-sided p-value
    """
    d = np.abs(e1)**power - np.abs(e2)**power  # loss differential
    T = len(d)
    d_bar = np.mean(d)
    
    # Newey-West HAC variance estimator (bandwidth = horizon - 1)
    gamma_0 = np.var(d, ddof=1)
    bandwidth = max(1, horizon - 1)
    
    autocovariances = 0.0
    for k in range(1, bandwidth + 1):
        weight = 1.0 - k / (bandwidth + 1)  # Bartlett kernel
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        autocovariances += 2 * weight * gamma_k
    
    var_d = (gamma_0 + autocovariances) / T
    
    if var_d <= 0:
        return 0.0, 1.0
    
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


# ============================================================
# Configuration
# ============================================================

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model name patterns in folder names
MODEL_PATTERNS = {
    'DCRNN': 'dcrnn',
    'AGCRN': 'agcrn',
    'GWNet': 'gwnet',
    'STGIN': 'stgin',
    'STAEformer': 'staeformer',
}

DATASETS = ['metr-la', 'pems-bay']
HORIZONS = [3, 6, 12]

# Comparison pairs: (baseline_config, enhanced_config, label)
COMPARISONS = [
    ('NoAcc', 'Acc_SG', 'NoAcc vs Acc_SG'),
    ('NoAcc', 'Acc_NoSG', 'NoAcc vs Acc_NoSG'),
]


def find_model_folder(model_key, dataset, config, horizon):
    """Find model folder matching the pattern."""
    pattern = MODEL_PATTERNS[model_key]
    
    # Handle GWNet special naming
    if model_key == 'GWNet':
        # Try gwnet first, then gwnet_acc_enhanced_v14
        candidates = [
            f"{pattern}_{dataset}_{config}_LSTM_Q{horizon}",
            f"{pattern}_acc_enhanced_v14_{dataset}_{config}_LSTM_Q{horizon}",
        ]
    elif model_key == 'STGIN':
        candidates = [
            f"{pattern}_{dataset}_{config}_LSTM_Q{horizon}",
            f"{pattern}_noste_{dataset}_{config}_LSTM_Q{horizon}",
        ]
    else:
        candidates = [
            f"{pattern}_{dataset}_{config}_LSTM_Q{horizon}",
            f"{pattern}_{dataset}_{config}_Q{horizon}",
        ]
    
    for c in candidates:
        path = os.path.join(MODELS_DIR, c)
        pred_file = os.path.join(path, 'predictions_mph.npy')
        if os.path.exists(pred_file):
            return path
    return None


def load_errors(folder):
    """Load predictions and targets, compute per-sample MAE."""
    pred = np.load(os.path.join(folder, 'predictions_mph.npy'))
    targ = np.load(os.path.join(folder, 'targets_mph.npy'))
    # Shape: (samples, nodes, horizon, 1) -> flatten to (samples,) via mean over nodes/horizon
    errors = np.abs(pred - targ).mean(axis=(1, 2, 3))  # per-sample MAE
    return errors


def main():
    results = []
    
    print("=" * 70)
    print("Diebold-Mariano Test: Pairwise Forecast Comparison")
    print("=" * 70)
    
    for model_key in MODEL_PATTERNS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for base_config, enh_config, label in COMPARISONS:
                    base_folder = find_model_folder(model_key, dataset, base_config, horizon)
                    enh_folder = find_model_folder(model_key, dataset, enh_config, horizon)
                    
                    if base_folder is None or enh_folder is None:
                        continue
                    
                    try:
                        e_base = load_errors(base_folder)
                        e_enh = load_errors(enh_folder)
                        
                        # Ensure same length
                        n = min(len(e_base), len(e_enh))
                        e_base = e_base[:n]
                        e_enh = e_enh[:n]
                        
                        dm_stat, p_value = diebold_mariano_test(
                            e_base, e_enh, horizon=horizon, power=1
                        )
                        
                        mae_base = e_base.mean()
                        mae_enh = e_enh.mean()
                        improvement = (mae_base - mae_enh) / mae_base * 100
                        
                        sig = ''
                        if p_value < 0.001:
                            sig = '***'
                        elif p_value < 0.01:
                            sig = '**'
                        elif p_value < 0.05:
                            sig = '*'
                        else:
                            sig = 'n.s.'
                        
                        results.append({
                            'Model': model_key,
                            'Dataset': dataset.upper(),
                            'Q': horizon,
                            'Comparison': label,
                            'MAE_Base': round(mae_base, 3),
                            'MAE_Enh': round(mae_enh, 3),
                            'Improvement%': round(improvement, 1),
                            'DM_Stat': round(dm_stat, 3),
                            'p_value': f'{p_value:.2e}' if p_value < 0.001 else f'{p_value:.4f}',
                            'Sig': sig,
                            'N_samples': n,
                        })
                        
                        print(f"  {model_key:6s} {dataset:8s} Q={horizon:2d} {label:20s} | "
                              f"DM={dm_stat:+7.3f}  p={p_value:.2e}  {sig:4s} | "
                              f"Impr={improvement:+5.1f}%")
                    
                    except Exception as ex:
                        print(f"  SKIP {model_key} {dataset} Q={horizon} {label}: {ex}")
    
    if results:
        df = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, '07_dm_test_results.csv')
        df.to_csv(out_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {out_path}")
        print(f"Total comparisons: {len(results)}")
        
        sig_count = sum(1 for r in results if r['Sig'] != 'n.s.')
        print(f"Significant (p<0.05): {sig_count}/{len(results)}")
        print(f"{'=' * 70}")
    else:
        print("\nNo comparisons found. Check model folder paths.")


if __name__ == '__main__':
    main()
