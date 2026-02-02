import os
from pathlib import Path
from typing import Optional
from .global_configuration import LEARNING_RATE, PATIENCE, WEIGHT_DECAY, DECAY_RATE, USE_LR_SCHEDULER
from .losses import mape_loss, rmse_loss, mase_loss, smape_loss
from .evaluation_utils import evaluate_model_properly
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
# ⚡ Mixed Precision Training (Compatible with PyTorch 1.x and 2.x)
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.0+
    PYTORCH_2_PLUS = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch 1.x fallback
    PYTORCH_2_PLUS = False

# Regularization policy: use Adam weight_decay only (default). To enable explicit L2, set USE_EXPLICIT_L2=True
L2_COEFF: float = 0.0
USE_EXPLICIT_L2: bool = False

# Training Loop
def train(model, train_loader, valid_loader, adj, epochs: int, with_acceleration: bool, results_save_dir='results', model_save_dir: Optional[str] = None, device: str = 'cuda'):
    model = model.to(device)
    adj = torch.FloatTensor(adj).to(device)
    
    # ⚡ OPTIMIZED: Fused Adam (benchmark-aligned, not AdamW)
    # Using Adam (not AdamW) to match DCRNN, Graph WaveNet, STGIN papers
    try:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY,
            fused=True  # Single CUDA kernel (faster)
        )
    except:
        # Fallback if fused not available
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
    
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
    criterion = nn.L1Loss()
    best_valid_loss = float('inf')
    patience_counter = 0
    
    # ⚡ Mixed Precision Training - 25-35% speedup on RTX 3050/3090
    # PyTorch 1.x: GradScaler() | PyTorch 2.0+: GradScaler('cuda') - both work with no args
    scaler = GradScaler() if device == 'cuda' else None
    use_amp = (device == 'cuda')  # Only use AMP on GPU

    train_history = {
        'train_loss': [],'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_smape': [], 'val_smape': [],
    }

    for epoch in range(epochs):
        model.train()

        train_loss_total = 0.0
        train_rmse_total = 0.0
        train_smape_total = 0.0

        for batch_x, batch_y, batch_ste in tqdm(train_loader, desc='Training', disable=True):  # Disable progress bar
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P+Q, 64)

            optimizer.zero_grad()
            
            # ⚡ Mixed Precision Forward Pass
            if use_amp:
                # PyTorch 1.x: autocast() | PyTorch 2.0+: autocast(device_type='cuda')
                autocast_context = autocast(device_type=device) if PYTORCH_2_PLUS else autocast()
                with autocast_context:
                    y_pred = model(batch_x, adj, batch_ste)
                    l2_reg = 0.0
                    if USE_EXPLICIT_L2:
                        l2_reg = sum(torch.norm(param, 2) ** 2 for param in model.parameters())
                    batch_loss = criterion(y_pred, batch_y) + L2_COEFF * l2_reg
                    batch_rmse = rmse_loss(y_pred, batch_y)
                    batch_smape = smape_loss(y_pred, batch_y)
                
                # Mixed precision backward pass
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)  # Unscale before gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 training (CPU or when AMP disabled)
                y_pred = model(batch_x, adj, batch_ste)
                l2_reg = 0.0
                if USE_EXPLICIT_L2:
                    l2_reg = sum(torch.norm(param, 2) ** 2 for param in model.parameters())
                batch_loss = criterion(y_pred, batch_y) + L2_COEFF * l2_reg
                batch_rmse = rmse_loss(y_pred, batch_y)
                batch_smape = smape_loss(y_pred, batch_y)
                
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_total += batch_loss.item() * batch_x.size(0)
            train_rmse_total += batch_rmse.item() * batch_x.size(0)
            train_smape_total += batch_smape.item() * batch_x.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)
        train_rmse = train_rmse_total / len(train_loader.dataset)
        train_smape = train_smape_total / len(train_loader.dataset)

        train_history['train_loss'].append(train_loss)
        train_history['train_rmse'].append(train_rmse)
        train_history['train_smape'].append(train_smape)

        model.eval()
        val_loss_total = 0.0
        val_rmse_total = 0.0
        val_smape_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_ste in tqdm(valid_loader, desc='Validation', disable=True):  # Disable progress bar
                batch_x = batch_x.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
                batch_y = batch_y.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
                batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P+Q, 64)

                # ⚡ Mixed Precision Inference
                if use_amp:
                    # PyTorch 1.x: autocast() | PyTorch 2.0+: autocast(device_type='cuda')
                    autocast_context = autocast(device_type=device) if PYTORCH_2_PLUS else autocast()
                    with autocast_context:
                        y_pred = model(batch_x, adj, batch_ste)
                        batch_loss = criterion(y_pred, batch_y)
                        batch_rmse = rmse_loss(y_pred, batch_y)
                        batch_smape = smape_loss(y_pred, batch_y)
                else:
                    y_pred = model(batch_x, adj, batch_ste)
                    batch_loss = criterion(y_pred, batch_y)
                    batch_rmse = rmse_loss(y_pred, batch_y)
                    batch_smape = smape_loss(y_pred, batch_y)

                val_loss_total += batch_loss.item() * batch_x.size(0)
                val_rmse_total += batch_rmse.item() * batch_x.size(0)
                val_smape_total += batch_smape.item() * batch_x.size(0)

        val_loss = val_loss_total / len(valid_loader.dataset)
        val_rmse = val_rmse_total / len(valid_loader.dataset)
        val_smape = val_smape_total / len(valid_loader.dataset)

        train_history['val_loss'].append(val_loss)
        train_history['val_rmse'].append(val_rmse)
        train_history['val_smape'].append(val_smape)

        # Save best model first
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            patience_counter = 0
            # Only print when saving best model (important epochs)
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train MAE: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train sMAPE: {train_smape:.2f}%, "
                  f"Val MAE: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val sMAPE: {val_smape:.2f}%, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"✅ Saving best model with validation loss: {best_valid_loss:.4f}")

            results_root = Path(results_save_dir)
            model_save_path = results_root / f"{train_loader.dataset.data_name}_results"
            model_save_path = model_save_path / ('with_acceleration' if with_acceleration else 'without_acceleration')
            model_save_path = model_save_path / (model_save_dir or f'history_{train_loader.dataset.history}_horizon_{train_loader.dataset.horizon}')

            model_save_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(train_history).to_csv(model_save_path / 'train_history.csv', index=False)
            torch.save(model.state_dict(), model_save_path / 'best_model.pt')
        elif (epoch + 1) % 10 == 0:
            # Print every 10 epochs for progress tracking
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} triggered. No improvement in validation loss for {PATIENCE} epochs.")
                break

        # Step LR scheduler (ReduceLROnPlateau needs validation metric)
        if scheduler is not None:
            scheduler.step(valid_loss)
    return train_history

# Training Loop
def train_compact(model, train_loader, valid_loader, adj, epochs: int, with_acceleration: bool, results_save_dir='', model_save_dir: Optional[str] = None, device: str = 'cuda'):
    model = model.to(device)
    adj = torch.FloatTensor(adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
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
    else:
        scheduler = None
    criterion = nn.L1Loss()
    best_valid_loss = float('inf')
    patience_counter = 0

    train_history = {
        'train_loss': [],'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_smape': [], 'val_smape': [],
    }

    for epoch in tqdm(range(epochs)):
        model.train()

        train_loss_total = 0.0
        train_rmse_total = 0.0
        train_smape_total = 0.0

        for batch_x, batch_y, batch_ste in train_loader:
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P+Q, 64)

            optimizer.zero_grad()
            y_pred = model(batch_x, adj, batch_ste)

            l2_reg = 0.0
            if USE_EXPLICIT_L2:
                l2_reg = sum(torch.norm(param, 2) ** 2 for param in model.parameters())
            batch_loss = criterion(y_pred, batch_y) + L2_COEFF * l2_reg

            batch_rmse = rmse_loss(y_pred, batch_y)
            batch_smape = smape_loss(y_pred, batch_y)

            batch_loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += batch_loss.item() * batch_x.size(0)
            train_rmse_total += batch_rmse.item() * batch_x.size(0)
            train_smape_total += batch_smape.item() * batch_x.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)
        train_rmse = train_rmse_total / len(train_loader.dataset)
        train_smape = train_smape_total / len(train_loader.dataset)

        train_history['train_loss'].append(train_loss)
        train_history['train_rmse'].append(train_rmse)
        train_history['train_smape'].append(train_smape)

        model.eval()
        val_loss_total = 0.0
        val_rmse_total = 0.0
        val_smape_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_ste in valid_loader:
                batch_x = batch_x.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
                batch_y = batch_y.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
                batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P+Q, 64)

                y_pred = model(batch_x, adj, batch_ste)
                batch_loss = criterion(y_pred, batch_y)

                batch_rmse = rmse_loss(y_pred, batch_y)
                batch_smape = smape_loss(y_pred, batch_y)

                val_loss_total += batch_loss.item() * batch_x.size(0)
                val_rmse_total += batch_rmse.item() * batch_x.size(0)
                val_smape_total += batch_smape.item() * batch_x.size(0)

        val_loss = val_loss_total / len(valid_loader.dataset)
        val_rmse = val_rmse_total / len(valid_loader.dataset)
        val_smape = val_smape_total / len(valid_loader.dataset)

        train_history['val_loss'].append(val_loss)
        train_history['val_rmse'].append(val_rmse)
        train_history['val_smape'].append(val_smape)

        if val_loss < best_valid_loss and epoch > 3:
            best_valid_loss = val_loss
            patience_counter = 0
            print(f"Saving the best model with validation loss: {best_valid_loss:.4f}")

            results_root = Path(results_save_dir) if results_save_dir else Path('.')
            model_save_path = results_root / f"{train_loader.dataset.data_name}_results"
            model_save_path = model_save_path / ('with_acceleration' if with_acceleration else 'without_acceleration')
            model_save_path = model_save_path / (model_save_dir or f'history_{train_loader.dataset.history}_horizon_{train_loader.dataset.horizon}')

            model_save_path.mkdir(parents=True, exist_ok=True)
    # Logging
    print(f"Train MAE: {min(train_history['train_loss']):.4f}, Train RMSE: {min(train_history['train_rmse']):.4f}, Train sMAPE: {min(train_history['train_smape']):.2f}%, "f"Val MAE: {min(train_history['val_loss']):.4f}, Val RMSE: {min(train_history['val_rmse']):.4f}, Val sMAPE: {min(train_history['val_smape']):.2f}%, ")
    return train_history

def test(model, test_loader, adj_matrix, device='cuda'):
    model = model.to(device)
    adj_matrix = torch.FloatTensor(adj_matrix).to(device)

    criterion = nn.L1Loss()
    model.eval()
    
    # ⚡ Mixed Precision Inference
    use_amp = (device == 'cuda')

    loss_total = 0.0
    rmse_total = 0.0
    smape_total = 0.0

    with torch.no_grad():
        for batch_x, batch_y, batch_ste in tqdm(test_loader):
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P+Q, 64)

            # ⚡ Mixed Precision Inference
            if use_amp:
                # PyTorch 1.x: autocast() | PyTorch 2.0+: autocast(device_type='cuda')
                autocast_context = autocast(device_type=device) if PYTORCH_2_PLUS else autocast()
                with autocast_context:
                    y_pred = model(batch_x, adj_matrix, batch_ste)
                    batch_loss = criterion(y_pred, batch_y)
                    batch_rmse = rmse_loss(y_pred, batch_y)
                    batch_smape = smape_loss(y_pred, batch_y)
            else:
                y_pred = model(batch_x, adj_matrix, batch_ste)
                batch_loss = criterion(y_pred, batch_y)
                batch_rmse = rmse_loss(y_pred, batch_y)
                batch_smape = smape_loss(y_pred, batch_y)

            loss_total += batch_loss.item() * batch_x.size(0)
            rmse_total += batch_rmse.item() * batch_x.size(0)
            smape_total += batch_smape.item() * batch_x.size(0)
            

    loss = loss_total / len(test_loader.dataset)
    rmse = rmse_total / len(test_loader.dataset)
    smape = smape_total / len(test_loader.dataset)

    return loss, rmse, smape
  
  
def test_thresholded(model, test_loader, adj_matrix, threshold, device='cuda'):
    model = model.to(device)
    adj_matrix = torch.FloatTensor(adj_matrix).to(device)

    criterion = nn.L1Loss(reduction='none')  # No reduction to apply point-wise mask

    model.eval()

    mae_total = 0.0
    rmse_total = 0.0
    smape_total = 0.0
    valid_points = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_ste in tqdm(test_loader):
            batch_x = batch_x.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P, FEATURES)
            batch_y = batch_y.to(device).permute(0, 2, 1, 3)  # (batch, nodes, Q, FEATURES)
            batch_ste = batch_ste.to(device).permute(0, 2, 1, 3)  # (batch, nodes, P+Q, 64)

            # Predict for all samples
            y_pred = model(batch_x, adj_matrix, batch_ste)

            # Create point-wise mask for ground truth > 0.2
            mask = batch_y > threshold  # Shape: (batch, nodes, Q, FEATURES)
            if not mask.any():
                continue  # Skip batch if no valid points

            # Apply mask to select valid points
            batch_y_valid = batch_y[mask]  # Flattened valid ground truth points
            batch_y_pred_valid = y_pred[mask]  # Flattened valid predicted points

            if len(batch_y_valid) == 0:
                continue

            # Compute point-wise losses
            point_mae = criterion(batch_y_pred_valid, batch_y_valid)  # Shape: (n_valid_points,)
            point_rmse = (batch_y_pred_valid - batch_y_valid) ** 2  # Shape: (n_valid_points,)
            point_smape = 200 * torch.abs(batch_y_pred_valid - batch_y_valid) / (
                torch.abs(batch_y_valid) + torch.abs(batch_y_pred_valid) + 1e-8
            )  # Shape: (n_valid_points,)

            # Accumulate metrics
            valid_batch_points = mask.sum().item()
            mae_total += point_mae.sum().item()
            rmse_total += point_rmse.sum().item()
            smape_total += point_smape.sum().item()
            valid_points += valid_batch_points


    if valid_points == 0:
        print(f"Warning: No points met the threshold criteria (speed > {threshold}).")
        return {'mae': float('nan'), 'rmse': float('nan'), 'smape': float('nan')}

    # Compute average metrics over valid points
    mae = mae_total / valid_points
    rmse = torch.sqrt(torch.tensor(rmse_total / valid_points)).item()
    smape = smape_total / valid_points

    return {'mae': mae, 'rmse': rmse, 'smape': smape}