# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from data_process import get_dataloaders, apply_rcse
from model import get_model
from utils import sis_loss, evaluate
import os

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_type).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_loader = get_dataloaders(args.batch_size, args.data_dir, 'train')
    val_loader = get_dataloaders(args.batch_size, args.data_dir, 'test')  # Use test as val for simplicity

    is_sis = 'SiS' in args.model_type
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for He, y in train_loader:
            He, y = He.to(device), y.to(device)
            He = apply_rcse(He)
            if is_sis:
                # Dynamic masking for sensor adaptability
                ps = model.ws.softmax(0)
                mask = torch.bernoulli(ps.repeat(He.size(0), 1)).bool()
                He = He * mask.unsqueeze(-1)
            y_hat = model(He)
            if is_sis:
                loss = sis_loss(y_hat, y, model.ws, args.lambda_s, args.lambda_v)
            else:
                loss = nn.MSELoss()(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_mae, val_rmse, val_r2 = 0, 0, 0
            for He, y in val_loader:
                He, y = He.to(device), y.to(device)
                He = apply_rcse(He)
                y_hat = model(He)
                mae, rmse, r2 = evaluate(y_hat, y)
                val_mae += mae.item()
                val_rmse += rmse.item()
                val_r2 += r2.item()
            print(f'Val MAE: {val_mae / len(val_loader)}, RMSE: {val_rmse / len(val_loader)}, R2: {val_r2 / len(val_loader)}')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{args.model_type}.pth')