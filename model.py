# model.py
import torch
import torch.nn as nn

class BaseFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_features=128, f=51):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        raise NotImplementedError

class Inception1D(BaseFeatureExtractor):
    def __init__(self, in_channels=1, out_features=128, f=51):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=5, padding=2)
        )
        self.fc = nn.Linear(96 * f, out_features)  # 32*3 = 96

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        b3 = torch.relu(self.branch3(x))
        out = torch.cat([b1, b2, b3], dim=1)
        out = out.flatten(1)
        out = self.fc(out)
        return out

class MobileNet1D(BaseFeatureExtractor):
    def __init__(self, in_channels=1, out_features=128, f=51):
        super().__init__()
        def dw_conv(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, stride=stride),
                nn.BatchNorm1d(c_in),
                nn.ReLU(),
                nn.Conv1d(c_in, c_out, kernel_size=1),
                nn.BatchNorm1d(c_out),
                nn.ReLU()
            )
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            dw_conv(32, 64, 1),
            dw_conv(64, 128, 2),
            dw_conv(128, 128, 1),
            dw_conv(128, 256, 2)
        )
        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, f)
            out = self.layers(dummy)
            flat_size = out.view(1, -1).size(1)
        self.fc = nn.Linear(flat_size, out_features)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out = self.layers(x)
        out = out.flatten(1)
        out = self.fc(out)
        return out

class ResNet1D(BaseFeatureExtractor):
    def __init__(self, in_channels=1, out_features=128, f=51):
        super().__init__()
        def res_block(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Conv1d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(c_out)
            )
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = res_block(64, 64)
        self.layer2 = res_block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x) + x[:, :64, ::2] if x.size(1) == 64 else self.layer1(x)  # Simplified residual
        x = self.layer2(x)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x

class SiSModel(nn.Module):
    def __init__(self, backbone='MobileNet', f=51, fb=128, S=3, D=3):
        super().__init__()
        self.S = S
        self.D = D
        if backbone == 'InceptionNet':
            self.fe = Inception1D(f=f, out_features=fb)
        elif backbone == 'MobileNet':
            self.fe = MobileNet1D(f=f, out_features=fb)
        elif backbone == 'ResNet':
            self.fe = ResNet1D(f=f, out_features=fb)
        else:
            raise ValueError("Unknown backbone")
        self.fr = nn.Sequential(
            nn.Linear(S * fb, 256),
            nn.ReLU(),
            nn.Linear(256, S * D)
        )
        self.ws = nn.Parameter(torch.ones(S))

    def forward(self, He):
        batch = He.size(0)
        X = [self.fe(He[:, s, :]) for s in range(self.S)]
        X = torch.stack(X, dim=1)  # (batch, S, fb)
        X_flat = X.flatten(1)
        Y_hat_flat = self.fr(X_flat)
        Y_hat = Y_hat_flat.view(batch, self.S, self.D)
        return Y_hat

class NonSiSModel(nn.Module):
    def __init__(self, f=51, fb=128, S=3, D=3):
        super().__init__()
        self.fe = MobileNet1D(f=f, out_features=fb)  # Fixed to MobileNet
        self.fr = nn.Linear(S * fb, S * D)

    def forward(self, He):
        batch = He.size(0)
        X = [self.fe(He[:, s, :]) for s in range(3)]
        X = torch.stack(X, dim=1)
        X_flat = X.flatten(1)
        Y_hat_flat = self.fr(X_flat)
        Y_hat = Y_hat_flat.view(batch, 3, 3)
        return Y_hat

# Placeholder for LocLite and SSLUL (simplified or from references)
class LocLite(nn.Module):
    def __init__(self, f=51, S=3, D=3):
        super().__init__()
        # Simplified as a basic CNN, actual impl from [42]
        self.conv = nn.Sequential(
            nn.Conv1d(S, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * (f // 2), D)

    def forward(self, He):
        He = He.transpose(1,2)  # Treat sensors as channels
        out = self.conv(He)
        out = self.fc(out)
        return out.unsqueeze(1).repeat(1, 3, 1)  # Dummy repeat

class SSLUL(nn.Module):
    def __init__(self, f=51, S=3, D=3):
        super().__init__()
        # Simplified self-supervised, actual from [43]
        self.encoder = nn.Sequential(
            nn.Conv1d(S, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * f, 256)
        )
        self.head = nn.Linear(256, D)

    def forward(self, He):
        He = He.transpose(1,2)
        enc = self.encoder(He)
        out = self.head(enc)
        return out.unsqueeze(1).repeat(1, 3, 1)  # Dummy

def get_model(model_type, f=51, fb=128, S=3, D=3):
    if model_type == 'InceptionNet_SiS':
        return SiSModel('InceptionNet', f, fb, S, D)
    elif model_type == 'MobileNet_SiS':
        return SiSModel('MobileNet', f, fb, S, D)
    elif model_type == 'ResNet_SiS':
        return SiSModel('ResNet', f, fb, S, D)
    elif model_type == 'MobileNet_nonSiS':
        return NonSiSModel(f, fb, S, D)
    elif model_type == 'LocLite':
        return LocLite(f, S, D)
    elif model_type == 'SSLUL':
        return SSLUL(f, S, D)
    else:
        raise ValueError("Unknown model type")