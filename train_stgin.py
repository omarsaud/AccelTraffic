#!/usr/bin/env python3
"""
STGIN Training Script (Official Baseline)

Features:
- argparse support for flexible experiments  
- Automatic Q-specific data path resolution (Q3_enhanced, Q6_enhanced, Q12_enhanced)
- Dual outputs (normalized + denormalized metrics)
- Official STGIN architecture (2 STBlocks, single LSTM, 1 bridge layer)

Usage:
    python testing_withenhancement.py --Q 3 --batch_size 32 --epochs 10 --use_acceleration true
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import pickle
import platform
import argparse
from datetime import datetime

# Import set_seed from utils_misc
from utils.utils_misc import set_seed

# 🎲 SET FIXED RANDOM SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# ⚡ Enable GPU optimizations (same as testing.py)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✅ GPU optimizations enabled (TF32 + cuDNN benchmark)")

# Setup paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.stgin_model import STGIN
from preprocessing.stgin_data_loading import prepare_data, get_data_paths, load_data
from utils.global_configuration import (
    LEARNING_RATE, BATCH_SIZE, HIDDEN_DIM, WEIGHT_DECAY,
    DROPOUT, PATIENCE, DECAY_RATE, USE_LR_SCHEDULER
)
STGIN_LEARNING_RATE = LEARNING_RATE
from utils.evaluation_utils import evaluate_model_properly, get_normalization_stats, compute_speed_weight_thresholds

# Auto-detect optimal num_workers based on OS (same as testing.py)
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4

# ⚡ Mixed Precision Training (Compatible with PyTorch 1.x and 2.x)
import torch
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.0+
    PYTORCH_2_PLUS = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch 1.x fallback
    PYTORCH_2_PLUS = False

# ================================
# BENCHMARK-ALIGNED CONFIGURATION
# ================================
# Same hyperparameters as ablation_baseline.py for fair comparison
DATASET_NAME = "metr-la"  # or "pems-bay"
HISTORY = 12  # 60 minutes
HORIZONS = [3, 6, 12]  # Test all horizons
EPOCHS = 100  # ✅ Benchmark standard (all models)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: Hyperparameters are now printed AFTER argparse in main() to show actual values used


def train_stgin(model, train_loader, val_loader, adj, epochs, device,
                speed_mean, speed_std, save_dir=None,
                learning_rate=None, weight_decay=None):
    """
    Training loop for STGIN model.
    Uses L1Loss (MAE) for fair comparison with baseline.
    ⚡ WITH MIXED PRECISION
    ⚡ WITH MAPE TRACKING (denormalized per-epoch)
    🔧 WITH CHECKPOINT SAVING (saves best model during training)
    
    Args:
        speed_mean: Speed mean for denormalization (from normalization_params.json)
        speed_std: Speed std for denormalization (from normalization_params.json)
        save_dir: Directory to save best model checkpoint
    
    Returns:
        dict with training_time, epochs_completed, best_val_loss
    """
    import time
    start_time = time.time()
    
    model = model.to(device)
    adj = torch.FloatTensor(adj).to(device)
    
    # Use passed speed_mean/std (already loaded from Q-enhanced folder)
    print(f"Loaded Z-score denormalization params: mean={speed_mean:.2f} mph, std={speed_std:.2f} mph for MAPE tracking")
    
    # Collect all trainable parameters
    params_to_optimize = list(model.parameters())
    
    # Resolve optimizer hyperparameters (align with multi-model script)
    lr = learning_rate if learning_rate is not None else LEARNING_RATE
    wd = weight_decay if weight_decay is not None else WEIGHT_DECAY

    # ⚡ Mixed Precision Training - 25-35% speedup
    # PyTorch 1.x: GradScaler() | PyTorch 2.0+: GradScaler('cuda')
    use_amp = (device == 'cuda')
    scaler = GradScaler() if use_amp else None

    # Decide whether to use fused AdamW (only when not using AMP, same as multi-model script)
    use_fused = (device == 'cuda') and (scaler is None)

    if use_fused:
        try:
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=lr,
                weight_decay=wd,
                fused=True  # Single CUDA kernel (faster)
            )
            print("⚡ Using fused AdamW optimizer (no AMP)")
        except Exception:
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=lr,
                weight_decay=wd
            )
            print("Using regular AdamW optimizer (fused not available)")
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=lr,
            weight_decay=wd
        )
        if scaler is not None:
            print("⚡ Using AdamW with AMP (fused disabled for compatibility)")
        else:
            print("Using regular AdamW optimizer")

    # Gradient clipping to prevent instability
    MAX_GRAD_NORM = 1.0
    
    # Learning rate scheduler - use ReduceLROnPlateau for fair comparison
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        print(f"✅ Using ReduceLROnPlateau scheduler (reduces LR only when validation stagnates)")
    else:
        scheduler = None
    
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = PATIENCE
    
    # Track training history including MAPE
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_mph': [],
        'val_mape_pct': []
    }
    
    print(f"\nTraining with L1Loss (MAE) + MAPE Tracking")
    print(f"First forward pass will take 30-60s (CUDA kernel compilation - normal!)")
    
    # DIAGNOSTIC: Test a simple CUDA operation first
    import time
    print(f"Testing CUDA with simple operation...")
    start = time.time()
    test_tensor = torch.randn(100, 100, device=device)
    test_result = test_tensor @ test_tensor
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Simple CUDA operation took {elapsed:.2f}s")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = torch.tensor(0.0, device=device)  # Keep on GPU for efficiency
        
        for batch_x, batch_y, batch_ste in train_loader:
            if epoch == 0 and train_loss.item() == 0.0:
                print(f"First batch: Moving data to GPU...")
            
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)
            
            if epoch == 0 and train_loss.item() == 0.0:
                print(f"Data on GPU. Calling model forward pass...")
                print(f"   batch_x: {batch_x.shape}, adj: {adj.shape}, batch_ste: {batch_ste.shape}")
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            if use_amp:
                autocast_context = autocast(device_type=device) if PYTORCH_2_PLUS else autocast()
                with autocast_context:
                    pred = model(batch_x, adj, batch_ste)
                    loss = F.l1_loss(pred, batch_y)
                
                # Mixed precision backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 (CPU)
                pred = model(batch_x, adj, batch_ste)
                loss = F.l1_loss(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_optimize, MAX_GRAD_NORM)
                optimizer.step()
            
            # Accumulate loss on GPU (avoid .item() sync in loop)
            train_loss += loss.detach() * batch_x.size(0)
        
        # Convert to scalar only once per epoch
        train_loss = train_loss.item() / len(train_loader.dataset)
        
        # Validation with MAPE calculation
        # CRITICAL FIX: Set model to eval() mode for correct BatchNorm behavior
        # During validation: Use running statistics, no MC Dropout needed
        # MC Dropout is ONLY for final test evaluation, not per-epoch validation
        model.eval()  # Uses running mean/std for BatchNorm (correct!)
        val_loss = torch.tensor(0.0, device=device)  # Keep on GPU for efficiency
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_ste in val_loader:
                batch_x = batch_x.to(device).permute(0, 2, 1, 3)
                batch_y = batch_y.to(device).permute(0, 2, 1, 3)
                batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)
                
                # Mixed Precision Inference
                if use_amp:
                    autocast_context = autocast(device_type=device) if PYTORCH_2_PLUS else autocast()
                    with autocast_context:
                        pred = model(batch_x, adj, batch_ste)
                        loss = F.l1_loss(pred, batch_y)
                else:
                    pred = model(batch_x, adj, batch_ste)
                    loss = F.l1_loss(pred, batch_y)
                
                # Accumulate loss on GPU
                val_loss += loss.detach() * batch_x.size(0)
                
                # Collect predictions and targets for MAPE
                val_predictions.append(pred.cpu())
                val_targets.append(batch_y.cpu())
        
        # Convert to scalar only once per epoch
        val_loss = val_loss.item() / len(val_loader.dataset)
        
        # Calculate MAPE on denormalized values
        val_preds_tensor = torch.cat(val_predictions, dim=0)
        val_targs_tensor = torch.cat(val_targets, dim=0)
        
        # Denormalize (Z-score)
        val_preds_denorm = val_preds_tensor * speed_std + speed_mean
        val_targs_denorm = val_targs_tensor * speed_std + speed_mean
        
        # Calculate MAE (denormalized)
        val_mae_denorm = torch.abs(val_preds_denorm - val_targs_denorm).mean().item()
        
        # Calculate MAPE with threshold >= 5 mph
        abs_diff = torch.abs(val_preds_denorm - val_targs_denorm)
        abs_targets = torch.abs(val_targs_denorm)
        valid_mask = abs_targets >= 5.0
        
        if valid_mask.sum() > 0:
            val_mape = (100.0 * torch.mean(abs_diff[valid_mask] / abs_targets[valid_mask])).item()
        else:
            val_mape = 0.0
        
        # Store in history
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['val_mae_mph'].append(val_mae_denorm)
        train_history['val_mape_pct'].append(val_mape)
        
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae_denorm:.2f} mph, "
            f"Val MAPE: {val_mape:.2f}% | Time: {epoch_time:.2f}s"
        )
        
        # Step LR scheduler (ReduceLROnPlateau needs validation metric)
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # CRITICAL FIX: Save checkpoint when validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model checkpoint
            if save_dir is not None:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'train_loss': train_loss
                }
                checkpoint_path = save_dir / "best_checkpoint.pt"
                torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    epochs_completed = epoch + 1
    
    # Load best checkpoint before returning
    if save_dir is not None:
        checkpoint_path = save_dir / "best_checkpoint.pt"
        if checkpoint_path.exists():
            print(f"\n📥 Loading best model from epoch {best_epoch}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best model loaded (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"⏱️  Training time: {training_time/60:.2f} minutes ({epochs_completed} epochs)")
    best_idx = train_history['val_loss'].index(min(train_history['val_loss']))
    print(f"📊 Best Val MAE: {train_history['val_mae_mph'][best_idx]:.2f} mph, MAPE: {train_history['val_mape_pct'][best_idx]:.2f}%")
    print(f"📊 Final Val MAE: {train_history['val_mae_mph'][-1]:.2f} mph, MAPE: {train_history['val_mape_pct'][-1]:.2f}%\n")
    
    return {
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'epochs_completed': epochs_completed,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'best_val_mae_mph': train_history['val_mae_mph'][best_idx],
        'best_val_mape_pct': train_history['val_mape_pct'][best_idx],
        'train_history': train_history
    }


def evaluate_model(model, test_loader, adj, device, speed_mean, speed_std):
    """
    Evaluate STGIN model on test set.
    
    Args:
        speed_mean: Speed mean for denormalization (from normalization_params.json)
        speed_std: Speed std for denormalization (from normalization_params.json)
    
    Returns:
        dict with standard metrics (scaled and denormalized)
    """
    import time
    eval_start_time = time.time()
    
    model.eval()
    adj = torch.FloatTensor(adj).to(device)
    
    all_preds = []
    all_targets = []
    
    print("🔬 Evaluating model...")
    
    with torch.no_grad():
        for batch_x, batch_y, batch_ste in test_loader:
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)
            
            pred = model(batch_x, adj, batch_ste)
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Standard metrics using evaluation_utils
    from utils.evaluation_utils import calculate_denormalized_metrics, calculate_speed_stratified_metrics
    print(f"📐 Z-score denormalization: mean={speed_mean:.2f} mph, std={speed_std:.2f} mph")
    
    # Calculate metrics in scaled space
    mae_scaled = torch.abs(predictions - targets).mean().item()
    rmse_scaled = torch.sqrt(((predictions - targets) ** 2).mean()).item()
    
    # Calculate denormalized metrics
    denorm_metrics = calculate_denormalized_metrics(predictions, targets, speed_mean, speed_std)
    
    # Calculate speed-stratified metrics
    stratified_metrics = calculate_speed_stratified_metrics(predictions, targets, speed_mean, speed_std)
    
    # Print stratified analysis
    print(f"\n📊 Speed-Stratified Analysis:")
    for range_name, metrics in stratified_metrics.items():
        if metrics['count'] > 0:
            print(f"  {range_name.capitalize()} ({metrics['speed_range_mph']} mph): "
                  f"MAE={metrics['mae']:.2f}, sMAPE={metrics['smape']:.1f}% "
                  f"({metrics['count']} samples, {metrics['percentage']:.1f}%)")
    
    eval_time_sec = time.time() - eval_start_time
    
    return {
        'standard': {
            'scaled': {
                'mae': mae_scaled,
                'rmse': rmse_scaled,
            },
            'denormalized': denorm_metrics,
            'stratified': stratified_metrics,
            'eval_time_sec': eval_time_sec
        }
    }


def run_single_experiment(dataset_name, horizon, use_acceleration,
                         speed_df, accel_np, timestamps, adj_mx,
                         speed_mean, speed_std,
                         batch_size=32, epochs=10, learning_rate=None,
                         norm_path=None, data_dir=None):
    """
    Run a single STGIN experiment.
    
    Args:
        dataset_name: "metr-la" or "pems-bay"
        horizon: Q (3, 6, or 12)
        use_acceleration: True/False
        speed_df: Loaded speed data (DataFrame)
        accel_np: Loaded acceleration data (numpy array or None)
        timestamps: Loaded timestamps
        adj_mx: Loaded adjacency matrix
        batch_size: Batch size (default: 32)
        epochs: Number of epochs (default: 10)
        learning_rate: Learning rate (default: from global_configuration)
    
    Returns:
        Dictionary with results
    """
    # Create config name for folder: NoAcc/Acc_SG/Acc_NoSG + LSTM
    config_parts = []

    # Determine configuration and whether SG filter was applied to acceleration
    if use_acceleration:
        if data_dir and 'unfiltered' in str(data_dir):
            config_parts.append('Acc_NoSG')
            sg_filter_applied = False
        else:
            config_parts.append('Acc_SG')
            sg_filter_applied = True
    else:
        config_parts.append('NoAcc')
        sg_filter_applied = False

    config_parts.append('LSTM')
    config_name = '_'.join(config_parts)
    
    input_dim = 2 if use_acceleration else 1
    lr_val = learning_rate if learning_rate else LEARNING_RATE
    print(f"\n{'='*100}")
    print(f"🎯 {config_name} | Q={horizon} ({horizon*5}min) | {input_dim}D → Speed")
    print(f"   Epochs={epochs}, Batch={batch_size}, LR={lr_val}, Patience={PATIENCE}")
    print(f"   Save: models/stgin_{dataset_name}_{config_name}_Q{horizon}/")
    print(f"{'='*100}\n")
    
    # Prepare data loaders (EXACTLY like testing.py)
    train_loader, val_loader, test_loader, info = prepare_data(
        speed_data=speed_df,  # Pass DataFrame (NOT .values!)
        acceleration_data=accel_np if use_acceleration else None,
        timestamps=timestamps,
        data_name=dataset_name,
        history=HISTORY,
        horizon=horizon,
        with_acceleration=use_acceleration,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,  # Use auto-detected (0 for Windows, 4 for Linux)
        cache_ste=True,
        reuse_ste_cache=True  # ⚡ CRITICAL: Reuse STE cache across experiments!
    )
    
    print(f"✅ Data loaders created (train: {len(train_loader)} batches)")
    
    # Build model (official baseline)
    input_dim = 2 if use_acceleration else 1
    model = STGIN(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")

    # Create save directory
    save_dir = ROOT / "models" / f"stgin_{dataset_name}_{config_name}_Q{horizon}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    training_info = train_stgin(
        model,
        train_loader,
        val_loader,
        adj_mx,
        epochs,
        DEVICE,
        speed_mean,
        speed_std,
        save_dir=save_dir,
        learning_rate=lr_val,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Save model
    model_path = save_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save training history with MAPE
    train_history_df = pd.DataFrame(training_info['train_history'])
    train_history_path = save_dir / "train_history.csv"
    train_history_df.to_csv(train_history_path, index=False)
    print(f"💾 Training history (with MAPE) saved to: {train_history_path}")
    
    # Evaluate on test set
    print("\n🔬 Evaluating on test set...")
    results = evaluate_model(model, test_loader, adj_mx, DEVICE, speed_mean, speed_std)
    
    eval_time = results['standard'].get('eval_time_sec', 0.0)
    
    # Save test metrics
    test_metrics = {
        'dataset': dataset_name,
        'horizon_Q': horizon,
        'horizon_minutes': horizon * 5,
        'random_seed': RANDOM_SEED,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'use_acceleration': use_acceleration,
            'sg_filter': sg_filter_applied,
            'data_normalized': True
        },
        'normalized_metrics': {
            'mae': float(results['standard']['scaled']['mae']),
            'rmse': float(results['standard']['scaled']['rmse'])
        },
        'denormalized_metrics': {
            'mae_mph': float(results['standard']['denormalized']['mae']),
            'rmse_mph': float(results['standard']['denormalized']['rmse']),
            'smape_pct': float(results['standard']['denormalized']['smape']),
            'mape_pct': float(results['standard']['denormalized']['mape']),
            'mape_threshold_mph': 5.0,
            'mape_support_pct': float(results['standard']['denormalized']['mape_support_pct'])
        },
        'stratified_metrics': results['standard']['stratified'],
        'training_info': {
            'training_time_sec': float(training_info['training_time_minutes'] * 60),
            'training_time_min': float(training_info['training_time_minutes']),
            'epochs_completed': training_info['epochs_completed'],
            'best_val_mae_mph': float(training_info['best_val_mae_mph']),
            'best_val_mape_pct': float(training_info['best_val_mape_pct']),
            'max_epochs': EPOCHS,
            'early_stop_patience': PATIENCE,
            'loss_function': 'L1Loss (MAE)'
        },
        'model_info': {
            'model_params': total_params,
            'input_dim': input_dim,
            'hidden_dim': HIDDEN_DIM
        },
        'hyperparameters': {
            'optimizer': 'AdamW',
            'learning_rate': lr_val,
            'batch_size': BATCH_SIZE,
            'weight_decay': WEIGHT_DECAY,
            'dropout': DROPOUT,
            'lr_scheduler': 'ExponentialLR',
            'decay_rate': DECAY_RATE
        },
        'evaluation_info': {
            'eval_time_sec': eval_time,
            'device': str(DEVICE)
        }
    }
    test_metrics_path = save_dir / "test_metrics.json"
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"💾 Test metrics saved to: {test_metrics_path}")
    
    # Save data split information
    split_info = {
        'dataset': dataset_name,
        'random_seed': RANDOM_SEED,
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'test_ratio': 0.2,
        'note': 'Following STG4Traffic benchmark standard (70/10/20)'
    }
    split_info_path = save_dir / "split_info.json"
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"💾 Split info saved to: {split_info_path}")
    
    # Save config
    config = {
        'input_dim': input_dim,
        'hidden_dim': HIDDEN_DIM,
        'with_acceleration': use_acceleration,
        'horizon': horizon,
        'history': HISTORY,
        'dataset': dataset_name,
        'config_name': config_name,
        'random_seed': RANDOM_SEED,
        'optimizer': 'AdamW',
        'learning_rate': lr_val,
        'batch_size': BATCH_SIZE,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'model_params': total_params,
        'loss_function': 'L1Loss (MAE)',
        'lr_scheduler': 'ExponentialLR',
        'decay_rate': DECAY_RATE
    }
    config_path = save_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save predictions (normalized and denormalized)
    print("\n📊 Saving predictions...")
    model.eval()
    all_preds_list = []
    all_targets_list = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_ste in test_loader:
            batch_x = batch_x.to(DEVICE).permute(0, 2, 1, 3)
            batch_y = batch_y.to(DEVICE).permute(0, 2, 1, 3)
            batch_ste = batch_ste.to(DEVICE).permute(0, 2, 1, 3)
            
            pred = model(batch_x, torch.FloatTensor(adj_mx).to(DEVICE), batch_ste)
            
            all_preds_list.append(pred.cpu())
            all_targets_list.append(batch_y.cpu())
    
    # Concatenate all batches
    predictions_all = torch.cat(all_preds_list, dim=0).numpy()
    targets_all = torch.cat(all_targets_list, dim=0).numpy()
    
    # Save NORMALIZED predictions
    np.save(save_dir / "predictions_normalized.npy", predictions_all)
    np.save(save_dir / "targets_normalized.npy", targets_all)
    print(f"   ✅ Saved normalized predictions: predictions_normalized.npy")
    
    # Save DENORMALIZED predictions [mph]
    predictions_denorm = predictions_all * speed_std + speed_mean
    targets_denorm = targets_all * speed_std + speed_mean
    np.save(save_dir / "predictions_mph.npy", predictions_denorm)
    np.save(save_dir / "targets_mph.npy", targets_denorm)
    print(f"   ✅ Saved denormalized predictions [mph]: predictions_mph.npy")
    
    # FIXED (Step 1 - Q-specific norm params copy): Use norm_path from Q-enhanced folder
    import shutil
    if norm_path and norm_path.exists():
        shutil.copy(norm_path, save_dir / "normalization_params.json")
        # Convert to absolute path to avoid relative_to() error on Windows
        abs_norm_path = norm_path.resolve() if not norm_path.is_absolute() else norm_path
        try:
            rel_path = abs_norm_path.relative_to(ROOT)
            print(f"   ✅ Copied normalization_params.json from {rel_path}")
        except ValueError:
            # If relative_to fails, just print the filename
            print(f"   ✅ Copied normalization_params.json from {norm_path.name}")
    else:
        print(f"   ⚠️  WARNING: norm_path not provided or not found, skipping normalization_params.json copy")
    
    print(f"\n✅ All predictions saved to: {save_dir}")
    
    # Print results
    print(f"\n{'='*100}")
    print(f"RESULTS - {config_name} (Q={horizon})")
    print(f"{'='*100}")
    print("\n Standard Metrics:")
    print(f"  MAE (denorm):  {results['standard']['denormalized']['mae']:.2f} mph")
    print(f"  RMSE (denorm): {results['standard']['denormalized']['rmse']:.2f} mph")
    print(f"  sMAPE:         {results['standard']['denormalized']['smape']:.2f}%")
    print(f"  MAPE:          {results['standard']['denormalized']['mape']:.2f}% (threshold: {results['standard']['denormalized'].get('mape_threshold', 5)} mph, support: {results['standard']['denormalized']['mape_support_pct']:.1f}%)")
    
    # Print speed-stratified performance
    if 'stratified' in results['standard']:
        print(f"\n Performance by Traffic Condition:")
        for range_name in ['congestion', 'slow', 'moderate', 'fast']:
            metrics = results['standard']['stratified'].get(range_name, {})
            if metrics.get('count', 0) > 0:
                print(f"  {range_name.capitalize():12} ({metrics['speed_range_mph']:>8} mph): "
                      f"MAE={metrics['mae']:5.2f} mph, sMAPE={metrics['smape']:5.1f}%, "
                      f"MAPE={metrics['mape']:5.1f}% "
                      f"({metrics['count']:6,} samples, {metrics['percentage']:4.1f}%)")
    
    print(f"\n✅ Standard predictor")
    
    print("\n⚙️ Training Info:")
    print(f"  Epochs completed: {training_info['epochs_completed']}")
    print(f"  Training time: {training_info['training_time_minutes']:.1f} min")
    print(f"  Best val loss: {training_info['best_val_loss']:.4f}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Loss function: L1Loss (MAE) - Fair comparison with baseline")
    print(f"{'='*100}\n")
    
    result_dict = {
        'config': config_name,
        'horizon': horizon,
        'acceleration': use_acceleration,
        'model_parameters': total_params,
        'training_time_minutes': training_info['training_time_minutes'],
        'epochs_completed': training_info['epochs_completed'],
        'best_val_loss': training_info['best_val_loss'],
        **results['standard']['denormalized']
    }
    
    # JSON LOGGING: Save individual experiment result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = ROOT / "runs" / dataset_name / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    json_result = {
        "dataset": dataset_name,
        "horizon_steps": horizon,
        "horizon_minutes": horizon * 5,
        "seed": RANDOM_SEED,
        "with_acceleration": use_acceleration,
        "config_name": config_name,
        "MAE_mph": float(results['standard']['denormalized']['mae']),
        "RMSE_mph": float(results['standard']['denormalized']['rmse']),
        "sMAPE_pct": float(results['standard']['denormalized']['smape']),
        "MAPE_pct": float(results['standard']['denormalized']['mape']),
        "MAPE_support_pct": float(results['standard']['denormalized']['mape_support_pct']),
        "train_time_min": float(training_info['training_time_minutes']),
        "epochs_completed": int(training_info['epochs_completed']),
        "model_parameters": int(total_params)
    }
    
    json_path = runs_dir / f"{config_name}_Q{horizon}.json"
    with open(json_path, 'w') as f:
        json.dump(json_result, f, indent=2)
    print(f"\n💾 JSON result saved: {json_path}")
    
    result_dict['json_path'] = str(json_path)
    return result_dict


def main():
    """Main function with argparse support for STGIN training."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='STGIN Training (Official Baseline)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay'])
    parser.add_argument('--data_dir', type=str, default=None, help='Custom data directory')
    
    # Experiment
    parser.add_argument('--Q', type=int, default=3, choices=[3, 6, 12], help='Prediction horizon')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model
    parser.add_argument('--use_acceleration', type=lambda x: x.lower() == 'true', default=True)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    
    # Print configuration
    print("\n" + "="*100)
    print(f"🚀 STGIN EXPERIMENT - {args.dataset.upper()}")
    print("="*100)
    print(f"📊 Config: Q={args.Q}, seed={args.seed}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"🧠 Model: Acceleration={'✅' if args.use_acceleration else '❌'}")
    print("="*100)
    
    # Resolve data paths automatically
    print(f"\n📂 Resolving data paths...")
    paths = get_data_paths(
        dataset_name=args.dataset,
        horizon_Q=args.Q,
        data_dir=args.data_dir,
        with_acceleration=args.use_acceleration
    )
    
    # Load data
    print(f"\n📥 Loading data...")
    speed_df, accel_np, adj_mx, timestamps = load_data(
        speed_path=str(paths['speed']),
        acceleration_path=str(paths['accel']) if paths['accel'] else None,
        adj_pkl_path=str(paths['adj'])
    )
    
    print(f"✅ Data loaded: speed {speed_df.shape}, adj {adj_mx.shape}")
    if accel_np is not None:
        print(f"   Accel: {accel_np.shape}")
    
    # Load normalization params EXPLICITLY from Q-enhanced folder
    norm_path = paths['norm_params']
    if not norm_path.exists():
        raise RuntimeError(f"❌ CRITICAL: normalization_params.json not found at {norm_path}")
    
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    
    speed_mean = norm_params['speed_mean']
    speed_std = norm_params['speed_std']
    
    # FIXED (Step 1 verification): Print Q-enhanced path explicitly
    # Convert to absolute path first if it's relative
    abs_norm_path = norm_path if norm_path.is_absolute() else (ROOT / norm_path).resolve()
    try:
        rel_path = abs_norm_path.relative_to(ROOT)
        print(f"\n[DENORM] Using norm from: {rel_path}")
    except ValueError:
        # Fallback if path is outside ROOT
        print(f"\n[DENORM] Using norm from: {norm_path}")
    print(f"   mean={speed_mean:.4f}, std={speed_std:.4f}")
    print(f"✅ Norm params loaded:")
    if 'accel_normalization' in norm_params:
        print(f"   Accel: {norm_params['accel_normalization']}")
    if 'speed_normalization' in norm_params:
        print(f"   Speed: {norm_params['speed_normalization']}")
    
    # Run experiment
    print(f"\n{'='*100}")
    print(f"🎯 Starting experiment...")
    print(f"{'='*100}")
    
    result = run_single_experiment(
        dataset_name=args.dataset,
        horizon=args.Q,
        use_acceleration=args.use_acceleration,
        speed_df=speed_df,
        accel_np=accel_np,
        timestamps=timestamps,
        adj_mx=adj_mx,
        speed_mean=speed_mean,
        speed_std=speed_std,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        norm_path=norm_path,
        data_dir=args.data_dir
    )
    
    # Print final results
    print(f"\n{'='*100}")
    print(f"✅ EXPERIMENT COMPLETE")
    print(f"{'='*100}")
    print(f"\n📊 Final Results:")
    print(f"   Config: {result['config']}")
    print(f"   MAE (mph): {result['mae']:.2f}")
    print(f"   RMSE (mph): {result['rmse']:.2f}")
    print(f"   Training time: {result['training_time_minutes']:.1f} min")
    print(f"\n📁 Output: models/stgin_{args.dataset}_{result['config']}_Q{args.Q}/")
    print("="*100)


if __name__ == "__main__":
    main()
