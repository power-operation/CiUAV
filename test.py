# test.py
import torch
from data_process import get_dataloaders, apply_rcse
from model import get_model
from utils import evaluate

def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_type).to(device)
    model.load_state_dict(torch.load(f'models/{args.model_type}.pth'))
    model.eval()
    test_loader = get_dataloaders(args.batch_size, args.data_dir, 'test')

    mae, rmse, r2 = 0, 0, 0
    with torch.no_grad():
        for He, y in test_loader:
            He, y = He.to(device), y.to(device)
            He = apply_rcse(He)
            y_hat = model(He)
            m, rm, r = evaluate(y_hat, y)
            mae += m.item()
            rmse += rm.item()
            r2 += r.item()
    print(f'Test MAE: {mae / len(test_loader)}, RMSE: {rmse / len(test_loader)}, R2: {r2 / len(test_loader)}')