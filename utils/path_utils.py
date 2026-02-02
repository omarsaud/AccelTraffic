from __future__ import annotations
from pathlib import Path
import os
from typing import Optional, Tuple, Union

COMMON_ROOTS = [
    Path('.'),
    Path('data'),
    Path('..') / 'data',
    Path.cwd(),
    Path.cwd() / 'data',
    # Colab / Google Drive mounts
    Path('/content/drive/MyDrive'),
    Path('/content/drive/MyDrive/STGNN'),
    Path.home() / 'drive' / 'MyDrive',
]

COMMON_FILENAMES = {
    'speed': ['scaled_speed.h5', 'scaled_filtered_speed.h5', 'metr-la.h5', 'pems-bay.h5', 'speed.h5'],
    'accel': ['scaled_acceleration.npy', 'scaled_filtered_acceleration.npy', 'filtered_acceleration.npy', 'acceleration.npy'],
    'adj':   ['adj_mx.pkl', 'adj.pkl']
}

# Fallback recursive glob patterns if exact filenames are not present
RECURSIVE_PATTERNS = {
    'speed': ['**/*metr*la*.h5', '**/*pems*bay*.h5', '**/*speed*.h5', '**/*scaled*filtered*speed*.h5'],
    'accel': ['**/*accel*.npy', '**/*acc*.npy', '**/*filtered*acc*.npy'],
    'adj':   ['**/*adj*mx*.pkl', '**/*adj*.pkl']
}

def expand(p: Union[str, Path]) -> Path:
    """Expand user (~), env vars, and resolve absolute path without requiring existence."""
    if isinstance(p, Path):
        s = str(p)
    else:
        s = p
    s = os.path.expandvars(os.path.expanduser(s))
    return Path(s)


def ensure_exists(p: Union[str, Path], kind: str) -> Path:
    path = expand(p)
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    return path


def try_find(kind: str, preferred_root: Optional[Path] = None, dataset_name: Optional[str] = None) -> Optional[Path]:
    """Try to find a common filename for kind ('speed'|'accel'|'adj') in common roots.
    
    Args:
        kind: Type of file to find ('speed'|'accel'|'adj')
        preferred_root: Preferred root directory to search first
        dataset_name: Optional dataset name ('metr-la'|'pems-bay') to prioritize
    """
    roots = []
    if preferred_root is not None:
        roots.append(preferred_root)
        roots.append(preferred_root / 'data')
    roots.extend(COMMON_ROOTS)
    
    # Prioritize dataset-specific paths if dataset_name is provided
    if dataset_name is not None:
        for root in roots:
            # Check dataset-specific directory first
            dataset_dir = root / 'data' / dataset_name
            if dataset_dir.exists():
                for name in COMMON_FILENAMES.get(kind, []):
                    candidate = dataset_dir / name
                    if candidate.exists():
                        return candidate
                # Also check for dataset-specific files (e.g., metr-la.h5)
                if kind == 'speed':
                    dataset_file = dataset_dir / f"{dataset_name}.h5"
                    if dataset_file.exists():
                        return dataset_file
    
    # Fall back to general search
    for root in roots:
        for name in COMMON_FILENAMES.get(kind, []):
            candidate = root / name
            if candidate.exists():
                return candidate
            # also check nested data folders
            candidate2 = root / 'data' / name
            if candidate2.exists():
                return candidate2
            # CHECK DEEPER: data/metr-la/ and data/pems-bay/ subdirectories
            for dataset in ['metr-la', 'pems-bay']:
                candidate3 = root / 'data' / dataset / name
                if candidate3.exists():
                    return candidate3
        # Recursive fallback under common subtrees
        for subtree in [root, root / 'data', root / 'datasets']:
            if not subtree.exists():
                continue
            for pattern in RECURSIVE_PATTERNS.get(kind, []):
                for match in subtree.rglob(pattern):
                    if match.is_file():
                        return match
    return None


def auto_locate(speed: Optional[Union[str, Path]], accel: Optional[Union[str, Path]], adj: Optional[Union[str, Path]], preferred_root: Optional[Union[str, Path]] = None, dataset_name: Optional[str] = None) -> Tuple[Path, Path, Path]:
    """Resolve or auto-discover dataset component paths.

    If explicit paths are provided and exist, they are used. Otherwise search COMMON_ROOTS.
    
    Args:
        speed: Path to speed file or None to auto-detect
        accel: Path to acceleration file or None to auto-detect
        adj: Path to adjacency file or None to auto-detect
        preferred_root: Preferred root directory for search
        dataset_name: Optional dataset name ('metr-la'|'pems-bay') to prioritize
    """
    pref = expand(preferred_root) if preferred_root else None

    # Resolve speed
    sp_path = expand(speed) if speed else None
    if sp_path is None or not sp_path.exists():
        found = try_find('speed', pref, dataset_name)
        if found is None:
            raise FileNotFoundError(f"Could not locate a speed HDF5 file for dataset '{dataset_name}' (e.g., {dataset_name}.h5 or scaled_filtered_speed.h5)")
        sp_path = found

    # Resolve acceleration
    ac_path = expand(accel) if accel else None
    if ac_path is None or not ac_path.exists():
        found = try_find('accel', pref, dataset_name)
        if found is None:
            raise FileNotFoundError(f"Could not locate an acceleration NPY file for dataset '{dataset_name}' (e.g., scaled_filtered_acceleration.npy or acceleration.npy)")
        ac_path = found

    # Resolve adjacency
    adj_path = expand(adj) if adj else None
    if adj_path is None or not adj_path.exists():
        found = try_find('adj', pref, dataset_name)
        if found is None:
            raise FileNotFoundError(f"Could not locate an adjacency pickle for dataset '{dataset_name}' (e.g., adj_mx.pkl)")
        adj_path = found

    return sp_path, ac_path, adj_path
