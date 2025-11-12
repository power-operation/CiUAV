# main.py
import argparse
from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(description="CiUAV Indoor UAV Localization Project")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--model_type', type=str, default='MobileNet_SiS', 
                        choices=['InceptionNet_SiS', 'MobileNet_SiS', 'ResNet_SiS', 
                                 'MobileNet_nonSiS', 'LocLite', 'SSLUL'], 
                        help='Model type')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Data directory')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lambda_s', type=float, default=1.0, help='Lambda for sensor regularization')
    parser.add_argument('--lambda_v', type=float, default=1.0, help='Lambda for sample regularization')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)

if __name__ == '__main__':
    main()