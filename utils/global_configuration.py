"""
Global Configuration & GPU Optimizations
Applies to all scripts: quick_test.py, testing.py, colab notebooks
"""

import torch

# ================================
# ⚡ GPU OPTIMIZATIONS (RTX 30xx)
# ================================
if torch.cuda.is_available():
    # TF32: 10-20% speedup on RTX 30xx with no accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # cuDNN benchmark: Auto-select fastest convolution algorithm
    torch.backends.cudnn.benchmark = True
    
    print("✅ GPU optimizations enabled (TF32 + cuDNN benchmark)")

# ================================
# HYPERPARAMETERS
# ================================
# Optimized for ablation studies on RTX 3050 6GB

# ================================
# BENCHMARK-ALIGNED HYPERPARAMETERS
# ================================
# Aligned with comparison models (we train STGIN, compare with their results):
# - DCRNN (Li et al., 2018): batch=64, lr=0.01→decay, dropout=0, wd=0
# - Graph WaveNet (Wu et al., 2019): batch=64, lr=0.001, dropout=0.3, wd=0.0001
# - STGCN (Yu et al., 2018): batch=64, similar to DCRNN
# - STG4Traffic benchmark: 7:1:2 split, patience=15-20, max_epochs=100
#
# Additional features from STGIN paper (Zou et al., 2023):
# - Decay rate: 0.9 (for LR scheduler)
# - Grid search was used to find optimal hyperparameters

# ===== CONSENSUS FROM BENCHMARK MODELS =====
BATCH_SIZE = 64  # ✅ DCRNN, Graph WaveNet, STGCN (ALL use 64)
HIDDEN_DIM = 64  # ✅ Universal standard
LEARNING_RATE = 0.001  # ✅ Aligned with multi-model scripts (DCRNN, GWNet, AGCRN, STGIN)
WEIGHT_DECAY = 0.0001  # ✅ Graph WaveNet uses 0.0001
DROPOUT = 0.3  # ✅ Graph WaveNet, STGIN
EPOCHS = 100  # ✅ ALL benchmarks use 100
PATIENCE = 15  # Fixed: Reduced from 15 to 10 for faster early stopping

# ===== ADDITIONAL FROM STGIN PAPER =====
DECAY_RATE = 0.9  # LR decay rate (from STGIN Table 1)
USE_LR_SCHEDULER = True  # Enable learning rate decay

# ===== DATA SPLIT (STG4Traffic Standard) =====
TRAIN_RATIO = 0.7  # 70% training (STG4Traffic for speed data)
VALID_RATIO = 0.1  # 10% validation (STG4Traffic)
TEST_RATIO = 0.2  # 20% test (STG4Traffic)
