# utils.py
import torch

def sis_loss(y_hat, y, ws, lambda_s=1.0, lambda_v=1.0, v=None):
    l_pred = torch.mean((y_hat - y) ** 2)
    ws_norm = ws / (torch.sum(ws) + 1e-8)
    l_sor = lambda_s * torch.sum(ws_norm * torch.log(1 + 1 / (ws_norm + 1e-8)))
    if v is None:
        v = torch.ones(y.size(0)).to(y.device)  # Placeholder, optimize v if needed
    l_sam = lambda_v * torch.mean(torch.sum(v.unsqueeze(-1).unsqueeze(-1) * (y_hat - y) ** 2, dim=[1,2]) / y.size(1))
    return l_pred + l_sor + l_sam

def evaluate(y_hat, y):
    mae = torch.mean(torch.abs(y_hat - y))
    rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum((y - y_hat) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return mae, rmse, r2