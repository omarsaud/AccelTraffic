"""
Data loading utilities for STGIN model with STE caching
VERSION: 2.0 - Fixed pin_memory for Windows
"""
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from pathlib import Path
# Removed unnecessary STGIN import that caused circular dependency
import torch.nn as nn
import torch
try:
    from .global_configuration import BATCH_SIZE, HIDDEN_DIM, TRAIN_RATIO, VALID_RATIO
except Exception:
    from global_configuration import BATCH_SIZE, HIDDEN_DIM, TRAIN_RATIO, VALID_RATIO

# ADDED: Q-specific data path resolver
def get_data_paths(dataset_name, horizon_Q, data_dir=None, with_acceleration=True):
    """
    Resolve data paths based on horizon Q.
    Priority: custom data_dir > Q-specific enhanced dir > default dir
    
    Args:
        dataset_name: 'metr-la' or 'pems-bay'
        horizon_Q: 3, 6, or 12
        data_dir: Optional custom directory path
        with_acceleration: Whether acceleration data is needed
    
    Returns:
        dict with keys: 'speed', 'accel', 'adj', 'norm_params', 'data_dir'
    """
    root = Path(__file__).resolve().parent.parent
    
    # Custom data_dir takes priority
    if data_dir is not None:
        base_dir = Path(data_dir)
        print(f"🔧 Using custom data_dir: {base_dir}")
    else:
        # Use Q-specific enhanced directory if exists
        enhanced_dir = root / "data" / dataset_name / f"Q{horizon_Q}_enhanced"
        default_dir = root / "data" / dataset_name
        
        if enhanced_dir.exists():
            base_dir = enhanced_dir
            print(f"✅ Using Q-specific enhanced data: {base_dir}")
        else:
            base_dir = default_dir
            print(f"⚠️  Q{horizon_Q}_enhanced not found, using default: {base_dir}")
    
    # Construct paths - prefer .npy (enhanced) over .h5 (original)
    speed_npy = base_dir / "scaled_speed.npy"
    speed_h5 = base_dir / f"{dataset_name}.h5"
    
    paths = {
        'data_dir': base_dir,
        'speed': speed_npy if speed_npy.exists() else speed_h5,
        'adj': base_dir / "adj_mx.pkl",
        'norm_params': base_dir / "normalization_params.json"
    }
    
    # Handle acceleration data (optional)
    if with_acceleration:
        # Try .npy first (enhanced), then .h5 (old format)
        accel_npy = base_dir / "scaled_acceleration.npy"
        accel_h5 = base_dir / f"{dataset_name}_acceleration.h5"
        
        if accel_npy.exists():
            paths['accel'] = accel_npy
        elif accel_h5.exists():
            paths['accel'] = accel_h5
        else:
            paths['accel'] = None
            print(f"⚠️  No acceleration data found in {base_dir}")
    else:
        paths['accel'] = None
    
    # Print actual paths for verification
    print(f"📂 Data paths resolved:")
    print(f"   Speed: {paths['speed']}")
    print(f"   Accel: {paths['accel']}")
    print(f"   Adj: {paths['adj']}")
    print(f"   Norm params: {paths['norm_params']}")
    
    return paths

# Load dataset and adjacency matrix
def load_data(speed_path='metr-la.h5', acceleration_path='acceleration.npy', adj_pkl_path='adj_mx.pkl'):
    # FIXED: Support both .h5 and .npy for speed
    speed_path = Path(speed_path)
    if speed_path.suffix == '.npy':
        speed_data = pd.DataFrame(np.load(speed_path))
        print(f"✅ Loaded speed from .npy: {speed_path.name}")
    else:
        speed_data = pd.read_hdf(speed_path)  # Shape: (num_timestamps, num_sensors)
        print(f"✅ Loaded speed from .h5: {speed_path.name}")
    
    # Handle optional acceleration data (can be None for ablation studies)
    if acceleration_path is not None:
        accel_path = Path(acceleration_path)
        if accel_path.suffix == '.npy':
            acceleration_data = np.load(acceleration_path)  # Shape: (num_timestamps, num_sensors)
            print(f"✅ Loaded acceleration from .npy: {accel_path.name}")
        else:
            acceleration_data = pd.read_hdf(acceleration_path).values
            print(f"✅ Loaded acceleration from .h5: {accel_path.name}")
        assert speed_data.shape == acceleration_data.shape, "Speed and acceleration data shapes do not match"
    else:
        acceleration_data = None

    adj_data = pd.read_pickle(adj_pkl_path)  # Shape: (num_sensors, num_sensors)
    sensor_names, sensor_to_id, adj_matrix = adj_data

    # Verify adjacency matrix shape
    assert adj_matrix.shape == (speed_data.shape[1], speed_data.shape[1]), "Adjacency matrix shape mismatch"

    # Prefer the actual index if it's a DateTimeIndex; otherwise try to recover real timestamps.
    timestamps = None
    timestamp_source = None
    if isinstance(speed_data.index, pd.DatetimeIndex) and len(speed_data.index) == speed_data.shape[0]:
        timestamps = speed_data.index
        timestamp_source = f"speed_h5_index:{speed_path.name}" if speed_path.suffix != '.npy' else "speed_dataframe_index"
    elif speed_path.suffix == '.npy':
        candidate_dirs = [speed_path.parent]
        if speed_path.parent.parent != speed_path.parent:
            candidate_dirs.append(speed_path.parent.parent)

        # If running from a variant folder (e.g., "metr-la-unfiltered"), also try the base dataset folder.
        parent_name = speed_path.parent.name
        if parent_name.endswith('-unfiltered'):
            sibling = speed_path.parent.parent / parent_name.replace('-unfiltered', '')
            if sibling not in candidate_dirs:
                candidate_dirs.append(sibling)

        ts_loaded = None
        for d in candidate_dirs:
            ts_npy = d / 'timestamps.npy'
            if ts_npy.exists():
                try:
                    ts_arr = np.load(ts_npy, allow_pickle=False)
                    ts_loaded = pd.to_datetime(ts_arr)
                    if len(ts_loaded) == speed_data.shape[0]:
                        timestamps = pd.DatetimeIndex(ts_loaded)
                        timestamp_source = f"timestamps_npy:{ts_npy}"
                        print(f"✅ Loaded real timestamps from: {ts_npy}")
                        break
                except Exception as e:
                    print(f"⚠️  Failed to load timestamps from {ts_npy}: {e}")

        if timestamps is None:
            h5_candidates = ['metr-la.h5', 'pems-bay.h5', 'speed.h5']
            for d in candidate_dirs:
                for h5_name in h5_candidates:
                    h5_path = d / h5_name
                    if h5_path.exists():
                        try:
                            df = pd.read_hdf(h5_path)
                            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) == speed_data.shape[0]:
                                timestamps = df.index
                                timestamp_source = f"recovered_h5_index:{h5_path}"
                                print(f"✅ Recovered real timestamps from: {h5_path}")
                                break
                        except Exception as e:
                            print(f"⚠️  Failed to read timestamps from {h5_path}: {e}")
                if timestamps is not None:
                    break

    if timestamps is None:
        print("⚠️  WARNING: Using synthetic timestamps (STE may be misaligned).")
        first_timestamp = pd.to_datetime('2012-01-01 00:00:00')
        timestamps = pd.date_range(start=first_timestamp, periods=speed_data.shape[0], freq='5min')
        timestamp_source = "synthetic_2012_5min"

    # Debug: print timestamp provenance for STE verification
    try:
        ts0 = timestamps[0]
        ts1 = timestamps[-1]
        print(f"🕒 Timestamp source: {timestamp_source}")
        print(f"🕒 Timestamp range: {ts0}  ->  {ts1}  (T={len(timestamps)})")
    except Exception as e:
        print(f"⚠️  Failed to print timestamp debug info: {e}")

    return speed_data, acceleration_data, adj_matrix, timestamps

# Embedding Layer and STE Generator (unchanged)
class EmbeddingLayer(nn.Module):
    def __init__(self, num_categories, output_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_categories, output_dim)

    def forward(self, indices):
        return self.embedding(indices)

class STEGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super(STEGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class STGINDataset(torch.utils.data.Dataset):
    def __init__(self, speed_data, acceleration_data, timestamps, data_name, history, horizon, start_idx, end_idx, with_acceleration: bool, temporal_components=("minute", "day"), cache_ste: bool = True, cache_key: str = None):
        self.data_name = data_name
        self.with_acceleration = with_acceleration
        self.cache_ste = cache_ste
        
        # Coerce to numpy arrays and sanitize
        if hasattr(speed_data, 'values'):
            speed_np = speed_data.values
        else:
            speed_np = np.asarray(speed_data)
        
        # Handle acceleration data (can be None for ablation studies)
        if acceleration_data is not None:
            accel_np = np.asarray(acceleration_data)
            accel_np = np.nan_to_num(accel_np, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            accel_np = None
        
        # Basic NaN/Inf handling for speed
        speed_np = np.nan_to_num(speed_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.speed_data = speed_np
        self.acceleration_data = accel_np
        self.timestamps = timestamps

        self.history = history
        self.horizon = horizon
        self.total_steps = history + horizon
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_samples = end_idx - start_idx - self.total_steps + 1

        self.num_sensors = speed_data.shape[1]

        # Verify dataset shape (only if acceleration data exists)
        if acceleration_data is not None:
            assert self.num_sensors == acceleration_data.shape[1], "Number of sensors in acceleration data does not match"

        # Initialize embedding layers and STE generator (dynamic input dim)
        self.temporal_components = tuple(temporal_components)
        self.spatial_embed_layer = EmbeddingLayer(self.num_sensors, HIDDEN_DIM // 4)
        self.minute_embed_layer = EmbeddingLayer(12, HIDDEN_DIM // 4)
        self.hour_embed_layer = EmbeddingLayer(24, HIDDEN_DIM // 4)
        self.day_embed_layer = EmbeddingLayer(7, HIDDEN_DIM // 4)
        self.week_embed_layer = EmbeddingLayer(52, HIDDEN_DIM // 4)
        spatial_dim = HIDDEN_DIM // 4
        n_temporal_used = len(self.temporal_components)
        ste_in_dim = (1 + n_temporal_used) * spatial_dim
        self.ste_generator = STEGenerator(input_dim=ste_in_dim, hidden_dim=HIDDEN_DIM)
        
        # Pre-compute or reuse STE cache if enabled
        self.ste_cache = {}
        self.cache_key = cache_key
        
        if self.cache_ste:
            if cache_key and cache_key in _GLOBAL_STE_CACHE:
                # FIXED (Step 4 - Log reduction): Reuse existing cache silently
                cached_data = _GLOBAL_STE_CACHE[cache_key]
                self.temporal_cache = cached_data['temporal_cache']
                self.spatial_embeds = cached_data['spatial_embeds']
                # Silent reuse (already logged when first created)
            else:
                # Build new cache using all timestamps
                self._precompute_ste_cache(self.timestamps)

        if self.with_acceleration and self.acceleration_data is not None:
            # Stack speed and acceleration: (num_timestamps, num_nodes, 2)
            self.features = np.stack([self.speed_data, self.acceleration_data], axis=-1)
        else:
            # Use only speed data: (num_timestamps, num_nodes, 1)
            self.features = self.speed_data[:, :, np.newaxis]

    def _precompute_ste_cache(self, all_timestamps):
        """Pre-compute STE embeddings for faster training from all available timestamps."""
        
        # ⚡ OPTIMIZATION: Try to load from disk first
        if self.cache_key:
            cache_file = _CACHE_DIR / f"ste_{self.cache_key}_fp16.pt"
            if cache_file.exists():
                try:
                    # Load cache (weights_only=False for dict with custom objects)
                    cached = torch.load(cache_file, weights_only=False)
                    # ⚡ CRITICAL FIX: Convert FP16 → FP32 to match model dtype
                    self.temporal_cache = {k: v.float() for k, v in cached['temporal_cache'].items()}
                    self.spatial_embeds = cached['spatial_embeds'].float()
                    print(f"⚡ Loaded STE cache from disk: {cache_file.name} ({len(self.temporal_cache)} patterns)")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load cache from disk: {e}, rebuilding...")
        
        print("Pre-computing STE cache for faster training...")

        # Pre-compute spatial embeddings (same for all timesteps)
        sensor_ids = torch.arange(self.num_sensors, dtype=torch.long)
        spatial_embeds = self.spatial_embed_layer(sensor_ids)

        # Pre-compute temporal embeddings for all unique timestamps across the entire dataset
        unique_combinations = set()
        for t in all_timestamps:
            minute = int(t.minute // 5)
            hour = int(t.hour)
            day = int(t.dayofweek)
            week = int(getattr(t.isocalendar(), 'week', 1) - 1) % 52
            unique_combinations.add((minute, hour, day, week))
        
        # Pre-compute embeddings for all unique combinations
        temporal_cache = {}
        for minute, hour, day, week in unique_combinations:
            minute_embed = self.minute_embed_layer(torch.tensor([minute], dtype=torch.long))
            hour_embed = self.hour_embed_layer(torch.tensor([hour], dtype=torch.long))
            day_embed = self.day_embed_layer(torch.tensor([day], dtype=torch.long))
            week_embed = self.week_embed_layer(torch.tensor([week], dtype=torch.long))
            
            temporal_list = []
            if "minute" in self.temporal_components:
                temporal_list.append(minute_embed)
            if "hour" in self.temporal_components:
                temporal_list.append(hour_embed)
            if "day" in self.temporal_components:
                temporal_list.append(day_embed)
            if "week" in self.temporal_components:
                temporal_list.append(week_embed)
            
            temporal_embed = torch.cat(temporal_list, dim=-1) if temporal_list else minute_embed
            temporal_cache[(minute, hour, day, week)] = temporal_embed
        
        self.temporal_cache = temporal_cache
        self.spatial_embeds = spatial_embeds
        print(f"STE cache built with {len(temporal_cache)} unique temporal patterns")
        
        # ⚡ OPTIMIZATION: Save to disk for future runs (FP16 for smaller files)
        if self.cache_key:
            cache_file = _CACHE_DIR / f"ste_{self.cache_key}_fp16.pt"
            try:
                # Convert to FP16 to save disk space and loading time
                cache_to_save = {
                    'temporal_cache': {k: v.half() for k, v in temporal_cache.items()},
                    'spatial_embeds': spatial_embeds.half()
                }
                torch.save(cache_to_save, cache_file)
                print(f"💾 STE cache saved to disk: {cache_file.name} (FP16 format)")
            except Exception as e:
                print(f"⚠️  Failed to save cache to disk: {e}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        global_idx = self.start_idx + idx
        x = self.features[global_idx:global_idx + self.history]  # (P, nodes, FEATURES)
        y = self.features[global_idx + self.history:global_idx + self.total_steps, :, 0:1]  # (Q, nodes, FEATURES)
        chunk_timestamps = self.timestamps[global_idx:global_idx + self.total_steps]

        # Generate STE - use cache if available, otherwise compute on-the-fly
        if self.cache_ste and hasattr(self, 'temporal_cache'):
            # ⚡ OPTIMIZED: Vectorized lookup (no Python loop)
            spatial_embeds = self.spatial_embeds
            
            # Extract temporal keys in one pass (vectorized)
            keys = [(int(t.minute // 5), int(t.hour), int(t.dayofweek), 
                     int(getattr(t.isocalendar(), 'week', 1) - 1) % 52) 
                    for t in chunk_timestamps]
            
            # Stack all embeddings directly (faster than cat)
            # Note: cache values have shape (1, embed_dim), stack gives (total_steps, 1, embed_dim)
            temporal_embeds = torch.cat([self.temporal_cache[k] for k in keys], dim=0)  # (total_steps, embed_dim)
        else:
            # Slow path: compute on-the-fly (original method)
            sensor_ids = torch.arange(self.num_sensors, dtype=torch.long)
            spatial_embeds = self.spatial_embed_layer(sensor_ids)

            minutes = torch.tensor([int(t.minute // 5) for t in chunk_timestamps], dtype=torch.long)
            hours = torch.tensor([int(t.hour) for t in chunk_timestamps], dtype=torch.long)
            days = torch.tensor([int(t.dayofweek) for t in chunk_timestamps], dtype=torch.long)
            weeks = torch.tensor([int(getattr(t.isocalendar(), 'week', 1) - 1) % 52 for t in chunk_timestamps], dtype=torch.long)

            minute_embeds = self.minute_embed_layer(minutes)
            hour_embeds = self.hour_embed_layer(hours)
            day_embeds = self.day_embed_layer(days)
            week_embeds = self.week_embed_layer(weeks)

            temporal_list = []
            if "minute" in self.temporal_components:
                temporal_list.append(minute_embeds)
            if "hour" in self.temporal_components:
                temporal_list.append(hour_embeds)
            if "day" in self.temporal_components:
                temporal_list.append(day_embeds)
            if "week" in self.temporal_components:
                temporal_list.append(week_embeds)
            temporal_embeds = torch.cat(temporal_list, dim=-1) if len(temporal_list) > 0 else minute_embeds

        spatial_embeds_expanded = spatial_embeds.unsqueeze(0).expand(self.total_steps, self.num_sensors, -1)
        temporal_embeds_expanded = temporal_embeds.unsqueeze(1).expand(-1, self.num_sensors, -1)

        combined_embeds = torch.cat([spatial_embeds_expanded, temporal_embeds_expanded], dim=-1)

        ste_window = self.ste_generator(combined_embeds).detach()  # (TOTAL_STEPS, 207, HIDDEN_DIM)

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y, ste_window

def get_data_loader_shape_info(data_loader):
    dataiter = iter(data_loader)
    features, labels, ste = next(dataiter)
    return {
        'Features' : list(features.size()),
        'Labels': list(labels.size()),
        "STE": list(ste.shape)
    }


# Global cache to share across datasets (in-memory)
_GLOBAL_STE_CACHE = {}

# Global cache directory for persistent storage
_CACHE_DIR = Path("cache")
_CACHE_DIR.mkdir(exist_ok=True)

# Optimized prepare_data with shared STE cache
def prepare_data(speed_data, acceleration_data, timestamps, data_name, history: int, horizon: int, with_acceleration: bool, batch_size=BATCH_SIZE, num_workers=0, cache_ste: bool = True, reuse_ste_cache: bool = False):
    num_timestamps = speed_data.shape[0]
    train_size = int(TRAIN_RATIO * (num_timestamps - history + horizon + 1))
    valid_size = int(VALID_RATIO * (num_timestamps - history + horizon + 1))

    # ⚡ OPTIMIZATION: Fixed cache key - STE doesn't depend on with_acceleration!
    # Old: included with_acceleration (incorrect)
    # New: only data_name, history, horizon, num_timestamps
    cache_key = f"{data_name}_H{history}_Q{horizon}_{num_timestamps}"
    
    # Create first dataset to build the cache
    print(f"🏗️ Creating datasets with cache key: {cache_key}")
    train_dataset = STGINDataset(
        speed_data, acceleration_data, timestamps, data_name, history, horizon, 
        start_idx=0, end_idx=train_size, with_acceleration=with_acceleration, 
        cache_ste=cache_ste, cache_key=cache_key if reuse_ste_cache else None
    )
    
    # FIXED (Step 4 - Log reduction): Share cache and log only once
    if reuse_ste_cache and cache_ste:
        _GLOBAL_STE_CACHE[cache_key] = {
            'temporal_cache': train_dataset.temporal_cache,
            'spatial_embeds': train_dataset.spatial_embeds
        }
        print(f"⚡ STE cache: {cache_key} ({len(train_dataset.temporal_cache)} patterns) - ready for reuse")
    
    valid_dataset = STGINDataset(
        speed_data, acceleration_data, timestamps, data_name, history, horizon,
        start_idx=train_size, end_idx=train_size + valid_size, with_acceleration=with_acceleration,
        cache_ste=cache_ste, cache_key=cache_key if reuse_ste_cache else None
    )
    
    test_dataset = STGINDataset(
        speed_data, acceleration_data, timestamps, data_name, history, horizon,
        start_idx=train_size + valid_size, end_idx=num_timestamps, with_acceleration=with_acceleration,
        cache_ste=cache_ste, cache_key=cache_key if reuse_ste_cache else None
    )

    # Optimized DataLoader settings for Windows/Linux compatibility
    use_pin_memory = (num_workers > 0)  # Only use pin_memory with workers (avoids Windows deadlock)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(num_workers > 0),  # Only use pin_memory with workers (avoids Windows issues)
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(num_workers > 0),  # Only use pin_memory with workers (avoids Windows issues)
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    info = {
        "General Information": {
            "Data Name": data_name,
            "Num Sensors": speed_data.shape[1],
            "Num Timestamps": num_timestamps,
            "History": history,
            "Horizon": horizon,
            "Features": 2 if with_acceleration else 1,
            "Train Dataset Size": len(train_dataset),
            "Valid Dataset Size": len(valid_dataset),
            "Test Dataset Size": len(test_dataset),
            "Total Dataset Size": len(train_dataset) + len(valid_dataset) + len(test_dataset),
        },
        "Data Loaders Information": {
            # NOTE: Shape info fetch disabled to avoid early DataLoader iteration (causes delays)
            # "Shape Info" : get_data_loader_shape_info(train_loader),
            "Train Loader": {"batch size": train_loader.batch_size, "number of workers": train_loader.num_workers},
            "Valid Loader": {"batch size": valid_loader.batch_size, "number of workers": valid_loader.num_workers},
            "Test Loader": {"batch size": test_loader.batch_size, "number of workers": test_loader.num_workers},
        },
    }

    return train_loader, valid_loader, test_loader, info

def load_best_model(model_save_path, device, input_dim=1, hidden_dim=HIDDEN_DIM):
  """Loads the best saved model from the specified directory.

  Args:
    model_save_path: Path to the directory containing saved models.
    device: The device to load the model onto ('cuda' or 'cpu').
    input_dim: Model input channels (1 for speed only, 2 for speed+acc).
    hidden_dim: Hidden dimension used during training.

  Returns:
    The loaded model, or None if no model is found.
  """
  model_save_path = Path(model_save_path)
  model_path = model_save_path / 'best_model.pt'
  if model_path.exists():
    model = STGIN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded best model from {model_path}")
    return model
  else:
    print(f"No model found at {model_path}")
    return None
