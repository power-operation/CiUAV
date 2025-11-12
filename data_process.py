# data_process.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CSIDataset(Dataset):
    def __init__(self, input_file, label_file):
        self.inputs = np.load(input_file)  # (N, S, f) e.g., (30800, 3, 51)
        self.labels = np.load(label_file)  # (N, S, 3)
        self.inputs = torch.from_numpy(self.inputs).float()
        self.labels = torch.from_numpy(self.labels).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def get_dataloaders(batch_size, data_dir='dataset', mode='train'):
    if mode == 'train':
        input_file = f'{data_dir}/train/train_data.npy'
        label_file = f'{data_dir}/train/train_label.npy'
    else:
        input_file = f'{data_dir}/test/test_data.npy'
        label_file = f'{data_dir}/test/test_label.npy'
    dataset = CSIDataset(input_file, label_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))

# RCSE Functions
def dac_compensation(Ho, agc_gain_db=20.0):
    rho = 10 ** (-agc_gain_db / 20.0)
    var_ho = torch.var(Ho, dim=-1, keepdim=True)
    mean_pow = torch.mean(Ho ** 2, dim=-1, keepdim=True)
    gamma = 1 - var_ho / (mean_pow + 1e-8)
    gamma = torch.clamp(gamma, min=0.0, max=1.0)
    H_hat = gamma * Ho / rho
    return H_hat

def ansor(H_hat, window_size=5, eta=3.0):
    # Simplified Hampel + low-pass
    batch, S, f = H_hat.shape
    # Padding for unfold
    pad = window_size // 2
    H_padded = torch.nn.functional.pad(H_hat, (pad, pad), mode='reflect')
    H_unfold = H_padded.unfold(2, window_size, 1)  # (batch, S, f, w)
    med = torch.median(H_unfold, dim=-1).values  # (batch, S, f)
    mad = torch.median(torch.abs(H_unfold - med.unsqueeze(-1)), dim=-1).values
    z_score = eta * torch.abs(H_hat - med) / (mad + 1e-8)
    outliers = z_score > 3.0
    H_filtered = H_hat.clone()
    H_filtered[outliers] = med[outliers]
    # Low-pass: Gaussian filter approx with conv
    weights = torch.tensor([0.0545, 0.2442, 0.4026, 0.2442, 0.0545]).view(1, 1, -1).to(H_filtered.device)
    # Reshape to (batch * S, 1, f) for conv1d, apply, then reshape back
    H_filtered = torch.conv1d(H_filtered.view(batch * S, 1, f), weights, padding=2).view(batch, S, f)
    
    return H_filtered

def apply_rcse(batch_inputs):
    H_hat = dac_compensation(batch_inputs)
    He = ansor(H_hat)
    return He