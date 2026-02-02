"""
Simple data loading utilities for multi-model system (DCRNN, GWNet, AGCRN)
No STE embeddings required - just raw speed and acceleration data.
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def load_data_simple(dataset_name, data_dir=None, use_acceleration=True):
    """
    Load data for multi-model system (simple interface, no STE).
    
    Args:
        dataset_name: 'metr-la' or 'pems-bay'
        data_dir: Optional custom data directory
        use_acceleration: Whether to load acceleration data
    
    Returns:
        speed_data: (num_timestamps, num_nodes)
        accel_data: (num_timestamps, num_nodes) or None
        adj_matrix: (num_nodes, num_nodes)
    """
    # Determine data directory
    if data_dir is not None:
        base_dir = Path(data_dir)
    else:
        root = Path(__file__).resolve().parent.parent
        base_dir = root / "data" / dataset_name
    
    print(f"📂 Loading data from: {base_dir}")
    
    # Load speed data (try .npy first, then .h5)
    speed_npy = base_dir / "scaled_speed.npy"
    speed_h5 = base_dir / f"{dataset_name}.h5"
    
    if speed_npy.exists():
        speed_data = np.load(speed_npy)
        print(f"✅ Loaded speed from .npy: {speed_data.shape}")
    elif speed_h5.exists():
        speed_data = pd.read_hdf(speed_h5).values
        print(f"✅ Loaded speed from .h5: {speed_data.shape}")
    else:
        raise FileNotFoundError(f"No speed data found in {base_dir}")
    
    # Load acceleration data (optional)
    accel_data = None
    if use_acceleration:
        accel_npy = base_dir / "scaled_acceleration.npy"
        accel_h5 = base_dir / f"{dataset_name}_acceleration.h5"
        
        if accel_npy.exists():
            accel_data = np.load(accel_npy)
            print(f"✅ Loaded acceleration from .npy: {accel_data.shape}")
        elif accel_h5.exists():
            accel_data = pd.read_hdf(accel_h5).values
            print(f"✅ Loaded acceleration from .h5: {accel_data.shape}")
        else:
            print(f"⚠️  No acceleration data found, using speed only")
    
    # Load adjacency matrix
    adj_path = base_dir / "adj_mx.pkl"
    if adj_path.exists():
        with open(adj_path, 'rb') as f:
            _, _, adj_matrix = pickle.load(f, encoding='latin1')
        print(f"✅ Loaded adjacency matrix: {adj_matrix.shape}")
    else:
        raise FileNotFoundError(f"No adjacency matrix found at {adj_path}")
    
    # Load normalization parameters
    norm_params = {}
    norm_path = base_dir / "normalization_params.json"
    if norm_path.exists():
        import json
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)
        print(f"✅ Loaded normalization parameters")
        print(f"   Speed: mean={norm_params.get('speed_mean', 'N/A'):.2f}, std={norm_params.get('speed_std', 'N/A'):.2f}")
    else:
        print(f"⚠️  No normalization parameters found, metrics will be in normalized scale")
    
    return speed_data, accel_data, adj_matrix, norm_params


class TrafficDataset(Dataset):
    """
    Simple dataset for multi-model system.
    
    Returns sliding windows of (historical, future) pairs.
    """
    
    def __init__(self, speed_data, accel_data, seq_len, horizon, start_idx, end_idx):
        """
        Args:
            speed_data: (num_timestamps, num_nodes)
            accel_data: (num_timestamps, num_nodes) or None
            seq_len: Historical window length
            horizon: Prediction horizon
            start_idx: Start index for this split
            end_idx: End index for this split
        """
        self.speed_data = speed_data
        self.accel_data = accel_data
        self.seq_len = seq_len
        self.horizon = horizon
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Number of valid samples
        self.num_samples = end_idx - start_idx - seq_len - horizon + 1
        
        # Determine input dimension
        if accel_data is not None:
            self.input_dim = 2  # Speed + acceleration
        else:
            self.input_dim = 1  # Speed only
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            x: (nodes, seq_len, input_dim) - historical data
            y: (nodes, horizon, 1) - future speed only
        """
        global_idx = self.start_idx + idx
        
        # Get speed data
        speed_hist = self.speed_data[global_idx:global_idx + self.seq_len]  # (seq_len, nodes)
        speed_future = self.speed_data[global_idx + self.seq_len:global_idx + self.seq_len + self.horizon]
        
        # Stack features
        if self.accel_data is not None:
            accel_hist = self.accel_data[global_idx:global_idx + self.seq_len]
            # Stack: (seq_len, nodes, 2)
            x = np.stack([speed_hist, accel_hist], axis=-1)
        else:
            # (seq_len, nodes, 1)
            x = speed_hist[:, :, np.newaxis]
        
        # Target is always speed only: (horizon, nodes, 1)
        y = speed_future[:, :, np.newaxis]
        
        # Transpose to (nodes, seq_len, input_dim) and (nodes, horizon, 1)
        x = np.transpose(x, (1, 0, 2))  # (nodes, seq_len, input_dim)
        y = np.transpose(y, (1, 0, 2))  # (nodes, horizon, 1)
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_data_loaders(speed_data, accel_data, seq_len, horizon, batch_size=32, 
                       train_ratio=0.7, val_ratio=0.1, num_workers=4):
    """
    Create train/val/test data loaders with optimizations.
    
    ⚡ OPTIMIZATIONS:
    - num_workers=4: Parallel data loading (30-50% speedup)
    - pin_memory=True: Faster GPU transfer (10-20% speedup)
    - prefetch_factor=2: Prefetch batches (5-10% speedup)
    - persistent_workers=True: Keep workers alive (5% speedup)
    
    Args:
        speed_data: (num_timestamps, num_nodes)
        accel_data: (num_timestamps, num_nodes) or None
        seq_len: Historical window
        horizon: Prediction horizon
        batch_size: Batch size
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        num_workers: Number of data loading workers (default: 4)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    import torch
    import platform
    
    # Determine optimal DataLoader settings
    use_pin_memory = torch.cuda.is_available()  # Only pin on GPU
    use_workers = (num_workers > 0)
    
    # Platform-specific optimizations
    if platform.system() == 'Windows' and use_workers:
        # Windows: num_workers > 0 can cause severe slowdowns with CUDA!
        # Force num_workers=0 on Windows to avoid 30+ minute hangs
        print("⚠️  Windows detected: Setting num_workers=0 (multiprocessing causes CUDA hangs)")
        num_workers = 0
        use_workers = False
    elif platform.system() == 'Linux' and use_workers:
        # Linux: Can use more workers for maximum CPU utilization
        # RTX 3090 can handle 8-16 workers efficiently
        if num_workers < 8:
            num_workers = 8  # Increase for better CPU utilization
            print(f"⚡ Linux detected: Using {num_workers} workers for maximum CPU utilization")
    
    num_timestamps = speed_data.shape[0]
    num_samples = num_timestamps - seq_len - horizon + 1
    
    # Calculate split indices
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    train_end = train_size
    val_end = train_size + val_size
    
    print(f"📊 Data splits:")
    print(f"   Total samples: {num_samples}")
    print(f"   Train: {train_size} ({train_ratio*100:.0f}%)")
    print(f"   Val: {val_size} ({val_ratio*100:.0f}%)")
    print(f"   Test: {num_samples - val_end} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Create datasets
    train_dataset = TrafficDataset(speed_data, accel_data, seq_len, horizon, 0, train_end)
    val_dataset = TrafficDataset(speed_data, accel_data, seq_len, horizon, train_end, val_end)
    test_dataset = TrafficDataset(speed_data, accel_data, seq_len, horizon, val_end, num_samples)
    
    # Create optimized data loaders
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
    }
    
    # Add prefetch and persistent workers if using multiprocessing
    if use_workers:
        dataloader_kwargs['prefetch_factor'] = 4  # ⚡ Prefetch 4 batches per worker (more aggressive)
        dataloader_kwargs['persistent_workers'] = True  # Keep workers alive between epochs
        dataloader_kwargs['drop_last'] = True  # ⚡ Drop incomplete batches for consistent GPU utilization
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# Test function
if __name__ == '__main__':
    print("Testing simple data loading...")
    
    # Test loading
    speed, accel, adj, norm_params = load_data_simple('metr-la', use_acceleration=True)
    print(f"\nLoaded data:")
    print(f"  Speed: {speed.shape}")
    print(f"  Accel: {accel.shape if accel is not None else None}")
    print(f"  Adj: {adj.shape}")
    print(f"  Norm params: {norm_params}")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        speed, accel, seq_len=12, horizon=3, batch_size=32
    )
    
    # Test batch
    x, y = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input (batch, nodes, seq_len, features): {x.shape}")
    print(f"  Target (batch, nodes, horizon, 1): {y.shape}")
    
    print("\n✅ Simple data loading works!")
