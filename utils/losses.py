import torch

def rmse_loss(y_pred, y_true):
    """Calculate RMSE between predictions and true values."""
    mse = torch.mean((y_pred - y_true) ** 2)
    return torch.sqrt(mse)

def mape_loss(y_pred, y_true, eps=1e-8):
    """Calculate MAPE, avoiding division by zero."""
    ape = torch.abs((y_true - y_pred) / (y_true + eps))
    return 100.0 * torch.mean(ape)

def smape_loss(y_pred, y_true, eps=1e-8):
    """Calculate sMAPE, symmetric MAPE."""
    abs_diff = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred) + eps) / 2
    return 100.0 * torch.mean(abs_diff / denominator)

def mase_loss(y_pred, y_true, eps=1e-8):
    """Calculate MASE, scaled by naive forecast error."""
    # FIXED: Vectorized computation on same device (no torch.tensor wrapper)
    naive_error = torch.mean(torch.abs(y_true[..., 1:, :] - y_true[..., :-1, :]))
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae / (naive_error + eps)