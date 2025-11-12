# CiUAV Project README

## Overview
This project implements the CiUAV system for indoor UAV localization using Channel State Information (CSI), as described in the paper "CiUAV: Scalable Device-Free Indoor UAV Localization via Multi-Objective Optimized Network Using Channel State Information". If reference is made, please cite this paper.

## Structure
- `main.py`: Entry point, handles train/test modes.
- `data_process.py`: Data loading and RCSE preprocessing.
- `utils.py`: Loss functions and evaluation metrics.
- `train.py`: Training loop.
- `test.py`: Testing loop.
- `model.py`: Model definitions with various backbones (InceptionNet, MobileNet, ResNet with SiS; non-SiS MobileNet; placeholders for LocLite and SSLUL).

## Requirements
- Python 3.10.16
- PyTorch 2.5.1
- NumPy
- ......


## Data
The example data is in the /dataset/ directory. Due to the large size of the dataset, we have uploaded a portion of the reference data, which is sufficient for the program to function properly. The data was collected in cooperation with the Power Equipment System Industry  Research Institution of FZU, the data is confidential. If you need cooperation, please email us. If you want to replace your own data, you can make changes according to the form the given datset file in /dataset/. Email address: CKDformer@163.com

Place data in `train/` and `test/` folders:
- `train_data.npy`: CSI data (N, 3, 51)
- `train_label.npy`: Labels (N, 3, 3)

## Usage
Train: `python main.py --mode train --model_type MobileNet_SiS`
Test: `python main.py --mode test --model_type MobileNet_SiS`
