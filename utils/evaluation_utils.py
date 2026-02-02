"""
Evaluation utilities for proper metric calculation
VERSION: 2.0 - Added caching for normalization stats

Adds comprehensive metrics in both scaled and denormalized spaces:
- MAE, RMSE, sMAPE, MAPE
"""
import torch
import torch.nn as nn
import numpy as np

# Speeds below this threshold (mph) are excluded from MAPE to avoid exploding percentages
MAPE_DENORM_MIN_SPEED_MPH = 5.0

# Cache for normalization stats to avoid repeated file I/O and prints
_NORMALIZATION_CACHE = {}

def denormalize_predictions(predictions, targets, speed_mean, speed_std):
    """
    Denormalize predictions and targets from Z-score back to original scale
    
    Args:
        predictions: Model predictions in Z-score normalized scale
        targets: Ground truth in Z-score normalized scale  
        speed_mean: Original mean speed value
        speed_std: Original std speed value
    
    Returns:
        denorm_preds, denorm_targets: Values in original scale (mph)
    """
    # Convert from Z-score to original scale: x_mph = x_norm * std + mean
    denorm_preds = predictions * speed_std + speed_mean
    denorm_targets = targets * speed_std + speed_mean
    
    return denorm_preds, denorm_targets

def calculate_denormalized_metrics(predictions, targets, speed_mean, speed_std):
    """
    Calculate metrics on denormalized (original scale) data
    
    Returns:
        dict with MAE, RMSE, sMAPE, MAPE in original units (mph)
    """
    # Denormalize to original scale (Z-score)
    denorm_preds, denorm_targets = denormalize_predictions(
        predictions, targets, speed_mean, speed_std
    )
    
    # Calculate metrics in original scale
    mae = torch.mean(torch.abs(denorm_preds - denorm_targets)).item()
    rmse = torch.sqrt(torch.mean((denorm_preds - denorm_targets) ** 2)).item()
    
    # sMAPE calculation (percentage)
    abs_diff = torch.abs(denorm_targets - denorm_preds)
    denominator = (torch.abs(denorm_targets) + torch.abs(denorm_preds)) / 2
    smape = (100.0 * torch.mean(abs_diff / (denominator + 1e-8))).item()

    # MAPE calculation (percentage) with masking of very low speeds to avoid numerically huge values
    with torch.no_grad():
        mask = torch.abs(denorm_targets) >= MAPE_DENORM_MIN_SPEED_MPH
        if mask.any():
            mape = (100.0 * torch.mean(abs_diff[mask] / (torch.abs(denorm_targets[mask]) + 1e-8))).item()
            mape_support_pct = (100.0 * mask.float().mean()).item()
        else:
            # Fallback if all targets are below threshold
            mape = (100.0 * torch.mean(abs_diff / (torch.abs(denorm_targets) + 1e-8))).item()
            mape_support_pct = 0.0
    
    return {
        'mae': mae,
        'rmse': rmse, 
        'smape': smape,
        'mape': mape,
        'mape_support_pct': mape_support_pct,
    }

def compute_speed_weight_thresholds(targets_denorm):
    """
    Compute data-driven speed thresholds for weighted loss.
    
    Returns thresholds based on quantiles of the target distribution:
    - Low-speed threshold (25th percentile): speeds below this get higher weight
    - Medium-speed threshold (50th percentile): speeds below this get medium weight
    
    This ensures the weighted loss adapts to different datasets automatically.
    
    Args:
        targets_denorm: Denormalized target speeds (in original scale, e.g., mph)
    
    Returns:
        tuple: (low_threshold, medium_threshold)
               Example: (35.2, 52.1) for METR-LA, or (42.5, 58.3) for PEMS-BAY
    """
    low_threshold = torch.quantile(targets_denorm, 0.25).item()
    medium_threshold = torch.quantile(targets_denorm, 0.50).item()
    return low_threshold, medium_threshold

def calculate_speed_stratified_metrics(predictions, targets, speed_mean, speed_std):
    """
    Calculate metrics stratified by speed ranges (DATA-DRIVEN quantile-based analysis)
    
    IMPORTANT: Speed thresholds are computed from the target data distribution,
    not hard-coded. This makes the analysis adaptive to different datasets
    (METR-LA, PEMS-BAY, or future datasets with different speed distributions).
    
    Strategy:
    - Congestion: 0 to 25th percentile (slowest 25% of traffic)
    - Slow: 25th to 50th percentile
    - Moderate: 50th to 75th percentile  
    - Fast: 75th to 100th percentile (fastest 25%)
    
    Returns:
        dict with metrics for each speed range
    """
    # Denormalize to original scale
    denorm_preds, denorm_targets = denormalize_predictions(
        predictions, targets, speed_mean, speed_std
    )
    
    # DATA-DRIVEN: Compute speed thresholds from target distribution
    # For large tensors (Q=12), use NumPy which handles memory better
    denorm_targets_np = denorm_targets.cpu().numpy().flatten()
    q25 = float(np.percentile(denorm_targets_np, 25))
    q50 = float(np.percentile(denorm_targets_np, 50))
    q75 = float(np.percentile(denorm_targets_np, 75))
    speed_max = float(denorm_targets_np.max())
    
    # Define speed ranges (mph) - ADAPTIVE to dataset
    ranges = {
        'congestion': (0, q25),           # Slowest 25%
        'slow': (q25, q50),               # 25-50th percentile
        'moderate': (q50, q75),           # 50-75th percentile
        'fast': (q75, speed_max + 0.1),   # Fastest 25%
    }
    
    stratified_metrics = {}
    
    for range_name, (min_speed, max_speed) in ranges.items():
        # Mask for this speed range
        mask = (denorm_targets >= min_speed) & (denorm_targets < max_speed)
        
        if mask.sum() == 0:
            # No samples in this range
            stratified_metrics[range_name] = {
                'count': 0,
                'percentage': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'smape': 0.0
            }
            continue
        
        # Extract samples in this range
        preds_range = denorm_preds[mask]
        targets_range = denorm_targets[mask]
        
        # Calculate metrics
        mae = torch.mean(torch.abs(preds_range - targets_range)).item()
        rmse = torch.sqrt(torch.mean((preds_range - targets_range) ** 2)).item()
        
        # sMAPE
        abs_diff = torch.abs(preds_range - targets_range)
        abs_sum = torch.abs(preds_range) + torch.abs(targets_range)
        smape = (200.0 * torch.mean(abs_diff / (abs_sum + 1e-8))).item()
        
        # MAPE (with threshold)
        mape_mask = torch.abs(targets_range) >= MAPE_DENORM_MIN_SPEED_MPH
        if mape_mask.sum() > 0:
            mape = (100.0 * torch.mean(
                torch.abs(preds_range[mape_mask] - targets_range[mape_mask]) / 
                torch.abs(targets_range[mape_mask])
            )).item()
        else:
            mape = 0.0
        
        stratified_metrics[range_name] = {
            'count': mask.sum().item(),
            'percentage': 100.0 * mask.sum().item() / targets.numel(),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'speed_range_mph': f'{min_speed}-{max_speed}'
        }
    
    return stratified_metrics

def get_normalization_stats(dataset_name='metr-la'):
    """
    Get Z-score normalization statistics for denormalization
    
    Loads actual parameters from your preprocessing (saved by extract_acceleration scripts)
    Uses cache to avoid repeated file I/O and print statements
    """
    # Check cache first (avoid repeated file I/O)
    if dataset_name in _NORMALIZATION_CACHE:
        return _NORMALIZATION_CACHE[dataset_name]
    
    import json
    from pathlib import Path
    
    # Try to load ACTUAL saved parameters from extract_acceleration_*.py
    param_file = Path(__file__).parent.parent / 'data' / dataset_name.lower() / 'normalization_params.json'
    
    # ENFORCED: normalization_params.json MUST exist (no fallback)
    if not param_file.exists():
        raise RuntimeError(
            f"❌ CRITICAL: normalization_params.json not found at {param_file}\n"
            f"   This file is REQUIRED for proper denormalization.\n"
            f"   Please run the preprocessing script first:\n"
            f"   - For METR-LA: python extract_acceleration_METR-LA_FINAL.py\n"
            f"   - For PEMS-BAY: python extract_acceleration_PEMSBAY_FINAL.py"
        )
    
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
            
            # Check if it's Z-score or old min-max format
            # Support both old field name (normalization_method) and new (speed_normalization)
            is_zscore = (
                ('normalization_method' in params and params['normalization_method'] == 'z-score') or
                ('speed_normalization' in params and params['speed_normalization'] == 'z-score')
            )
            
            if is_zscore:
                # Only print on first load (not on cache hits)
                print(f"✅ Loaded Z-score normalization params from {param_file.name}: mean={params['speed_mean']:.2f}, std={params['speed_std']:.2f}")
                
                # Return ALL parameters (not just speed_mean/std)
                # This preserves all metadata for validation, reporting, and thesis documentation
                result = {
                    # Speed normalization (USED for denormalization)
                    'speed_mean': params['speed_mean'],
                    'speed_std': params['speed_std'],
                    'speed_min': params.get('speed_min', 0.0),
                    'speed_max': params.get('speed_max', 100.0),
                    'speed_normalization': params.get('speed_normalization', 'z-score'),
                    'normalization_method': 'z-score',
                    
                    # SG filter metadata (for validation & reporting)
                    'sg_filtered': params.get('sg_filtered', False),
                    'sg_window': params.get('sg_window', None),
                    'sg_poly': params.get('sg_poly', None),
                    
                    # Acceleration normalization (for denormalization & validation)
                    'accel_min_original': params.get('accel_min_original', None),
                    'accel_max_original': params.get('accel_max_original', None),
                    'accel_normalization': params.get('accel_normalization', None),
                    'accel_scaled_range': params.get('accel_scaled_range', None),
                    'accel_clip_threshold': params.get('accel_clip_threshold', None),
                    'accel_method': params.get('accel_method', None),
                    
                    # Documentation metadata
                    'scenario': params.get('scenario', 'unknown'),
                    'description': params.get('description', ''),
                }
                
                # Cache the result to avoid repeated loads
                _NORMALIZATION_CACHE[dataset_name] = result
                return result
            else:
                # Old min-max format - convert to approximate Z-score
                print(f"⚠️ Warning: Old min-max normalization detected. Please re-run extract_acceleration scripts!")
                speed_min = params.get('speed_min', 0.0)
                speed_max = params.get('speed_max', 76.67)
                # Approximate: assume normal distribution
                speed_mean = (speed_min + speed_max) / 2
                speed_std = (speed_max - speed_min) / 4
                print(f"⚠️ Using approximate Z-score: mean={speed_mean:.2f}, std={speed_std:.2f}")
                return {
                    'speed_mean': speed_mean,
                    'speed_std': speed_std,
                    'speed_min': speed_min,
                    'speed_max': speed_max,
                    'normalization_method': 'approximate-z-score'
                }
    except Exception as e:
        raise RuntimeError(
            f"❌ CRITICAL: Failed to parse normalization_params.json at {param_file}\n"
            f"   Error: {e}\n"
            f"   Please check the file format or re-run preprocessing."
        )

def evaluate_model_properly(model, test_loader, adj, device, speed_mean: float, speed_std: float):
    """
    Evaluate model with proper denormalization
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        adj: Adjacency matrix
        device: Device (cuda/cpu)
        speed_mean: Speed mean for denormalization (from normalization_params.json)
        speed_std: Speed std for denormalization (from normalization_params.json)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    # Ensure adj is a tensor and on the correct device
    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj).float().to(device)
    else:
        adj = adj.to(device)
    
    with torch.no_grad():
        for batch_x, batch_y, batch_ste in test_loader:
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)  
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)
            
            y_pred = model(batch_x, adj, batch_ste)
            
            all_predictions.append(y_pred.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # FIXED (dual outputs): Calculate BOTH normalized and denormalized metrics
    # 1) NORMALIZED metrics (Z-score space)
    abs_diff_norm = torch.abs(predictions - targets)
    denom_smape_norm = (torch.abs(targets) + torch.abs(predictions)) / 2
    normalized_metrics = {
        'MAE_norm': torch.mean(abs_diff_norm).item(),
        'RMSE_norm': torch.sqrt(torch.mean((predictions - targets) ** 2)).item(),
        'sMAPE_norm': (100.0 * torch.mean(abs_diff_norm / (denom_smape_norm + 1e-8))).item(),
    }
    
    # 2) DENORMALIZED metrics (mph space)
    denorm_metrics = calculate_denormalized_metrics(
        predictions, targets, 
        speed_mean, speed_std
    )
    
    # FIXED (dual outputs): Return both with clear naming
    return {
        'normalized': normalized_metrics,      # Z-score space
        'denormalized': denorm_metrics,        # mph space (includes MAPE>=5mph)
        # For backward compatibility
        'scaled': normalized_metrics,
    }


def evaluate_model_per_step(model, test_loader, adj, device, speed_mean: float, speed_std: float):
    """
    Evaluate model with PER-STEP metrics for detailed horizon analysis
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        adj: Adjacency matrix
        device: Device (cuda/cpu)
        speed_mean: Speed mean for denormalization (from normalization_params.json)
        speed_std: Speed std for denormalization (from normalization_params.json)
    
    Returns metrics for EACH prediction step separately:
    - Horizon 3 → metrics for step 1, 2, 3
    - Horizon 12 → metrics for step 1, 2, ..., 12
    
    This helps analyze:
    - How error propagates over prediction horizon
    - Which steps are hardest to predict
    - Where model improvements have most impact
    """
    model.eval()
    all_predictions = []
    all_targets = []

    # Ensure adj is a tensor and on the correct device
    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj).float().to(device)
    else:
        adj = adj.to(device)
    
    with torch.no_grad():
        for batch_x, batch_y, batch_ste in test_loader:
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)  
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)
            
            y_pred = model(batch_x, adj, batch_ste)
            
            all_predictions.append(y_pred.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all batches: shape (N, T, V, F)
    # N = samples, T = time steps (horizon), V = nodes, F = features
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    horizon = predictions.shape[1]  # Number of time steps
    
    # Compute metrics for each step
    per_step_results = []
    
    for step in range(horizon):
        # Extract predictions and targets for this specific step
        # shape: (N, V, F) - all samples, all nodes, for this one time step
        step_preds = predictions[:, step, :, :]
        step_targets = targets[:, step, :, :]
        
        # Scaled metrics
        abs_diff_scaled = torch.abs(step_preds - step_targets)
        denom_smape_scaled = (torch.abs(step_targets) + torch.abs(step_preds)) / 2
        scaled = {
            'mae': torch.mean(abs_diff_scaled).item(),
            'rmse': torch.sqrt(torch.mean((step_preds - step_targets) ** 2)).item(),
            'smape': (100.0 * torch.mean(abs_diff_scaled / (denom_smape_scaled + 1e-8))).item(),
        }
        
        # Denormalized metrics
        denorm = calculate_denormalized_metrics(
            step_preds, step_targets,
            speed_mean, speed_std
        )
        
        per_step_results.append({
            'step': step + 1,  # 1-indexed for readability
            'scaled': scaled,
            'denormalized': denorm
        })
    
    # Also compute overall aggregate (same as evaluate_model_properly)
    abs_diff_all = torch.abs(predictions - targets)
    denom_smape_all = (torch.abs(targets) + torch.abs(predictions)) / 2
    aggregate_scaled = {
        'mae': torch.mean(abs_diff_all).item(),
        'rmse': torch.sqrt(torch.mean((predictions - targets) ** 2)).item(),
        'smape': (100.0 * torch.mean(abs_diff_all / (denom_smape_all + 1e-8))).item(),
    }
    
    aggregate_denorm = calculate_denormalized_metrics(
        predictions, targets,
        speed_mean, speed_std
    )
    
    return {
        'per_step': per_step_results,  # List of dicts, one per step
        'aggregate': {
            'scaled': aggregate_scaled,
            'denormalized': aggregate_denorm
        },
        'horizon': horizon
    }


def print_preprocessing_summary(params):
    """
    Print a comprehensive summary of data preprocessing
    
    Args:
        params: Dictionary of normalization parameters (from get_normalization_stats)
    """
    print("\n" + "="*80)
    print("📊 DATA PREPROCESSING SUMMARY")
    print("="*80)
    
    # Scenario info
    if params.get('scenario'):
        print(f"Scenario: {params['scenario']}")
    if params.get('description'):
        print(f"Description: {params['description']}")
    
    # Speed processing
    print(f"\n🚄 Speed Processing:")
    if params.get('sg_filtered'):
        print(f"   ✅ SG Filter Applied:")
        print(f"      - Window size: {params.get('sg_window', 'N/A')}")
        print(f"      - Polynomial order: {params.get('sg_poly', 'N/A')}")
        print(f"      - Type: Causal (backward-looking)")
    else:
        print(f"   ⚠️  No SG filter applied (raw speed)")
    
    print(f"   ✅ Normalization: {params.get('speed_normalization', 'unknown')}")
    print(f"      - Mean: {params['speed_mean']:.2f} mph")
    print(f"      - Std: {params['speed_std']:.2f} mph")
    
    # Acceleration processing
    print(f"\n⚡ Acceleration Processing:")
    if params.get('accel_method'):
        print(f"   ✅ Computation Method: {params['accel_method']}")
    
    if params.get('accel_normalization'):
        print(f"   ✅ Normalization: {params['accel_normalization']}")
        if params.get('accel_scaled_range'):
            print(f"      - Target range: {params['accel_scaled_range']}")
        if params.get('accel_min_original') is not None:
            print(f"      - Original range: [{params['accel_min_original']:.4f}, {params['accel_max_original']:.4f}] mph/min")
        if params.get('accel_clip_threshold'):
            print(f"      - Outlier clipping: ±{params['accel_clip_threshold']:.4f} mph/min (99th percentile)")
    else:
        print(f"   ⚠️  No acceleration data or normalization info")
    
    print("="*80 + "\n")


def validate_preprocessing_params(params, expected=None):
    """
    Validate that preprocessing parameters match expected values
    
    Args:
        params: Loaded normalization parameters
        expected: Optional dict of expected values to validate against
    
    Returns:
        bool: True if valid, False otherwise
    """
    issues = []
    
    # Check essential fields exist
    required_fields = ['speed_mean', 'speed_std', 'speed_normalization']
    for field in required_fields:
        if field not in params or params[field] is None:
            issues.append(f"Missing required field: {field}")
    
    # Check if expected values match (if provided)
    if expected:
        for key, expected_val in expected.items():
            if key in params:
                actual_val = params[key]
                if actual_val != expected_val:
                    issues.append(f"{key}: expected {expected_val}, got {actual_val}")
    
    # Print validation result
    if issues:
        print("\n⚠️  PREPROCESSING VALIDATION FAILED:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ Preprocessing validation passed")
        return True


def denormalize_acceleration(accel_normalized, params):
    """
    Denormalize acceleration from normalized scale back to original mph/min
    
    Args:
        accel_normalized: Normalized acceleration values
        params: Normalization parameters dict
    
    Returns:
        Denormalized acceleration in mph/min
    """
    accel_norm_method = params.get('accel_normalization', 'unknown')
    
    if accel_norm_method == 'min-max':
        # Reverse Min-Max scaling: x_norm = 2(x - min)/(max - min) - 1
        # Solve for x: x = (x_norm + 1) * (max - min) / 2 + min
        accel_min = params['accel_min_original']
        accel_max = params['accel_max_original']
        accel_original = (accel_normalized + 1) * (accel_max - accel_min) / 2 + accel_min
        return accel_original
    
    elif accel_norm_method == 'z-score':
        # Reverse Z-score: x_norm = (x - mean) / std
        # Solve for x: x = x_norm * std + mean
        accel_mean = params.get('accel_mean', 0.0)
        accel_std = params.get('accel_std', 1.0)
        accel_original = accel_normalized * accel_std + accel_mean
        return accel_original
    
    else:
        print(f"⚠️  Warning: Unknown acceleration normalization method: {accel_norm_method}")
        print(f"   Returning normalized values unchanged")
        return accel_normalized
