#!/usr/bin/env python3
"""
Multi-Model Training Script for Traffic Prediction
===================================================

Supports multiple models with acceleration preprocessing framework:
- STGIN (Your enhanced version)
- DCRNN (Diffusion Convolutional RNN)
- Graph WaveNet (Adaptive graph learning)
- AGCRN (Adaptive Graph Convolutional RNN)

All models support 2-channel input (speed + acceleration).

Usage:
    python testing_multi_model.py --model stgin --dataset metr-la --Q 3 \
        --use_acceleration true --data_dir data/metr-la
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# ⚡ Import global configuration (enables TF32, cuDNN)
import utils.global_configuration  # This activates GPU optimizations on import

# ⚡ Import optimization utilities
try:
    from utils.model_optimizer import optimize_model_for_training, get_optimization_summary
    HAS_OPTIMIZER = True
except ImportError:
    HAS_OPTIMIZER = False
    print("⚠️  model_optimizer not found, running without torch.compile")

# Import model factory
from models.model_factory import create_model
from preprocessing.simple_data_loading import load_data_simple, create_data_loaders
# Note: Metrics are computed directly in evaluate() function

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # NOTE: We keep cuDNN benchmark ON for speed (set in global_configuration)
    # Only set deterministic=True for exact reproducibility
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Model Traffic Prediction')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['stgin', 'dcrnn', 'gwnet', 'agcrn'],
                       help='Model to train')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['metr-la', 'pems-bay'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Custom data directory (optional)')
    
    # Model configuration
    parser.add_argument('--use_acceleration', type=str, default='false',
                       choices=['true', 'false'],
                       help='Use acceleration as 2nd channel')
    
    # Prediction horizon
    parser.add_argument('--Q', type=int, default=3,
                       choices=[3, 6, 12],
                       help='Prediction horizon')
    parser.add_argument('--H', type=int, default=12,
                       help='Historical window')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model-specific hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3)

    # Optional output directory (used by SG sensitivity scripts)
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Custom output directory for saving model and results (optional)')
    
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, criterion, device, adj_matrix, scaler=None, first_epoch=False, model_name=''):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    adj_matrix_tensor = torch.FloatTensor(adj_matrix).to(device)
    
    total_batches = len(train_loader)
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # Print progress for first epoch (minimal - avoid CUDA sync from flush)
        if first_epoch and batch_idx == 0:
            print(f"⏳ Starting epoch (batch_size={x.shape[0]}, nodes={x.shape[1]})...")
        
        x = x.to(device)  # (batch, nodes, seq_len, input_dim)
        y = y.to(device)  # (batch, nodes, horizon, 1)
        
        optimizer.zero_grad()
        
        # Forward pass (clean - no debug prints)
        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(x, adj_matrix_tensor)
                loss = criterion(pred, y)
        else:
            pred = model(x, adj_matrix_tensor)
            loss = criterion(pred, y)
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, data_loader, device, adj_matrix, norm_params=None):
    """Evaluate model with proper denormalization"""
    model.eval()
    predictions = []
    targets = []
    
    adj_matrix = torch.FloatTensor(adj_matrix).to(device)
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x, adj_matrix)
            
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Initialize denormalized arrays
    predictions_denorm = None
    targets_denorm = None
    
    # Denormalize if parameters available
    if norm_params and 'speed_mean' in norm_params and 'speed_std' in norm_params:
        speed_mean = norm_params['speed_mean']
        speed_std = norm_params['speed_std']
        
        # Denormalize: x_real = x_norm * std + mean
        predictions_denorm = predictions * speed_std + speed_mean
        targets_denorm = targets * speed_std + speed_mean
        
        # Calculate metrics on denormalized data (real MPH values)
        mae = np.mean(np.abs(predictions_denorm - targets_denorm))
        rmse = np.sqrt(np.mean((predictions_denorm - targets_denorm) ** 2))
        
        # MAPE with threshold to avoid division by near-zero values
        mask = targets_denorm > 5.0  # Only speeds > 5 mph
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((predictions_denorm[mask] - targets_denorm[mask]) / targets_denorm[mask])) * 100
        else:
            mape = np.mean(np.abs((predictions_denorm - targets_denorm) / (targets_denorm + 1e-5))) * 100
    else:
        # Fallback: normalized metrics
        print("⚠️  Computing metrics on normalized data (no denormalization params)")
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / (np.abs(targets) + 1e-5))) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions,
        'targets': targets,
        'predictions_denorm': predictions_denorm if norm_params else None,
        'targets_denorm': targets_denorm if norm_params else None
    }


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert boolean flags
    use_acceleration = (args.use_acceleration.lower() == 'true')
    input_dim = 2 if use_acceleration else 1
    
    # Determine data directory
    if args.data_dir is None:
        args.data_dir = f'data/{args.dataset}'
    
    print("="*80)
    print(f"MULTI-MODEL TRAINING: {args.model.upper()}")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Acceleration: {'✅' if use_acceleration else '❌'}")
    print(f"Input dim: {input_dim}")
    print(f"Horizon (Q): {args.Q}")
    print(f"Historical (H): {args.H}")
    print("="*80)
    
    # Load data
    print("\n📥 Loading data...")
    speed_data, accel_data, adj, norm_params = load_data_simple(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        use_acceleration=use_acceleration
    )
    
    # Create optimized data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        speed_data=speed_data,
        accel_data=accel_data,
        seq_len=args.H,
        horizon=args.Q,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.1,
        num_workers=4  # ⚡ Parallel data loading (30-50% speedup)
    )
    
    num_nodes = speed_data.shape[1]
    print(f"✅ Data loaded: speed {speed_data.shape}, {num_nodes} nodes")
    
    # Create model
    print(f"\n🧠 Creating {args.model.upper()} model...")
    model = create_model(
        model_name=args.model,
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        historical_window=args.H,
        prediction_horizon=args.Q,
        dropout=args.dropout
    )
    # ⚡ Apply model optimizations (torch.compile, etc.)
    # NOTE: torch.compile disabled due to OOM (CUDA graphs use too much VRAM)
    if HAS_OPTIMIZER:
        model = optimize_model_for_training(model, device, enable_compile=False)  # Disabled to fix OOM
    else:
        model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created: {num_params:,} parameters")
    
    # Setup training with optimized optimizer
    criterion = nn.L1Loss()  # MAE loss
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # ⚡ Use AdamW with fused optimizer (5-10% speedup)
    # NOTE: fused doesn't work with AMP due to dtype conflicts
    use_fused = (device.type == 'cuda') and (scaler is None)  # Only if not using AMP
    
    if use_fused:
        try:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay,
                fused=True  # ⚡ Single CUDA kernel
            )
            print("⚡ Using fused AdamW optimizer (no AMP)")
        except:
            # Fallback to regular AdamW
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print("Using regular AdamW optimizer (fused not available)")
    else:
        # Use regular AdamW with AMP (more important than fused)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if scaler:
            print("⚡ Using AdamW with AMP (fused disabled for compatibility)")
        else:
            print("Using regular AdamW optimizer")
    
    # Use ReduceLROnPlateau for complex models (reduces LR only when validation stagnates)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # Print optimization summary
    if HAS_OPTIMIZER:
        get_optimization_summary()
    
    # Training loop
    print(f"\n🎯 Training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    training_start_time = time.time()
    training_history = []  # Track training history
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, adj, scaler, first_epoch=(epoch==1), model_name=args.model)
        
        # Validate
        val_results = evaluate(model, val_loader, device, adj, norm_params)
        val_mae = val_results['mae']
        
        # Learning rate schedule (ReduceLROnPlateau needs validation metric)
        scheduler.step(val_mae)
        
        epoch_time = time.time() - epoch_start
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae': val_mae,
            'val_rmse': val_results['rmse'],
            'val_mape': val_results['mape'],
            'epoch_time': epoch_time
        })
        
        # Display with proper units
        if norm_params and 'speed_mean' in norm_params:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} mph | "
                  f"Time: {epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            patience_counter = 0
            
            # Determine output directory for this run
            if args.output_dir is not None:
                # If caller (e.g., SG sensitivity script) specifies an explicit
                # output directory, respect it exactly.
                model_dir = Path(args.output_dir)
            else:
                # Backward-compatible default naming based on acceleration and
                # whether data_dir indicates SG-filtered vs unfiltered.
                if use_acceleration:
                    # Detect if using SG-filtered or unfiltered data from data_dir
                    if 'unfiltered' in args.data_dir:
                        config_name = 'Acc_NoSG_LSTM'
                    else:
                        config_name = 'Acc_SG_LSTM'
                else:
                    config_name = 'NoAcc_LSTM'
                
                model_dir = Path(f'models/{args.model}_{args.dataset}_{config_name}_Q{args.Q}')
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'best_model.pt')

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'val_mae': best_val_loss,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, model_dir / 'best_checkpoint.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    # Test
    print(f"\n🔬 Testing best model...")
    model.load_state_dict(torch.load(model_dir / 'best_model.pt'))
    test_results = evaluate(model, test_loader, device, adj, norm_params)
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    if norm_params and 'speed_mean' in norm_params:
        print(f"MAE:  {test_results['mae']:.2f} mph")
        print(f"RMSE: {test_results['rmse']:.2f} mph")
    else:
        print(f"MAE:  {test_results['mae']:.4f} (normalized)")
        print(f"RMSE: {test_results['rmse']:.4f} (normalized)")
    print(f"MAPE: {test_results['mape']:.2f}%")
    print("="*80)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    best_epoch_num = epoch - patience_counter
    
    # Save comprehensive results (aligned with STGIN format)
    results = {
        # Model info
        'model': args.model,
        'dataset': args.dataset,
        'use_acceleration': use_acceleration,
        'input_dim': input_dim,
        
        # Configuration
        'Q': args.Q,
        'H': args.H,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        
        # Test metrics (denormalized if possible)
        'test_mae': float(test_results['mae']),
        'test_rmse': float(test_results['rmse']),
        'test_mape': float(test_results['mape']),
        
        # Training info
        'num_params': num_params,
        'epochs_trained': best_epoch_num,
        'total_epochs': epoch,
        'train_time_seconds': float(total_training_time),
        'train_time_minutes': float(total_training_time / 60),
        
        # Best validation
        'best_val_mae': float(best_val_loss),
        
        # Normalization info
        'has_denormalization': bool(norm_params and 'speed_mean' in norm_params),
        'speed_mean': norm_params.get('speed_mean', None) if norm_params else None,
        'speed_std': norm_params.get('speed_std', None) if norm_params else None,
        
        # Timestamp
        'timestamp': datetime.now().isoformat()
    }
    
    with open(model_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save additional files (like STGIN)
    print(f"\n💾 Saving comprehensive results...")
    
    # 1. Save training history
    import pandas as pd
    train_df = pd.DataFrame(training_history)
    train_df.to_csv(model_dir / 'train_history.csv', index=False)
    print(f"✅ Saved training history")
    
    # 2. Save predictions and targets
    if test_results['predictions'] is not None:
        np.save(model_dir / 'predictions_normalized.npy', test_results['predictions'])
        np.save(model_dir / 'targets_normalized.npy', test_results['targets'])
        print(f"✅ Saved normalized predictions/targets")
    
    if test_results['predictions_denorm'] is not None:
        np.save(model_dir / 'predictions_mph.npy', test_results['predictions_denorm'])
        np.save(model_dir / 'targets_mph.npy', test_results['targets_denorm'])
        print(f"✅ Saved denormalized predictions/targets (MPH)")
    
    # 3. Save model configuration
    model_config = {
        'model_name': args.model,
        'architecture': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': 1,
            'dropout': args.dropout,
            'historical_window': args.H,
            'prediction_horizon': args.Q
        },
        'num_parameters': num_params,
        'dataset': args.dataset,
        'num_nodes': num_nodes
    }
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✅ Saved model configuration")
    
    # 4. Save normalization parameters
    if norm_params:
        with open(model_dir / 'normalization_params.json', 'w') as f:
            json.dump(norm_params, f, indent=2)
        print(f"✅ Saved normalization parameters")
    
    # 5. Save data split info
    split_info = {
        'total_samples': len(speed_data) - args.H - args.Q + 1,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'test_ratio': 0.2
    }
    with open(model_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✅ Saved data split info")
    
    print(f"\n✅ All results saved to: {model_dir}/")
    print(f"\n📊 Summary:")
    print(f"   Training time: {total_training_time/60:.1f} minutes")
    print(f"   Best epoch: {best_epoch_num}/{epoch}")
    print(f"   Total files saved: 11 (matching STGIN format)")
    print(f"\n Files:")
    for file in sorted(model_dir.glob('*')):
        print(f"     - {file.name}")


if __name__ == '__main__':
    main()

