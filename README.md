# CIFAR-10 Image Classification with ResNet

Implemented a ResNet-32 classifier for CIFAR-10 and compared two learning-rate settings.

## Overview
- **Task:** CIFAR-10 image classification
- **Model:** ResNet-32
- **Classes:** 10
- **Batch size:** 128
- **Optimizer:** SGD with Momentum
- **Weight decay:** 1e-4
- **Maximum epochs:** 200
- **Learning-rate schedule:** MultiStepLR at steps 32,000 and 48,000

## Reported Experiments

| Setting | Learning Rate | Accuracy | Training Time |
|---|---:|---:|---:|
| #1 | 0.1 | 92.27% | 60.44 min |
| #2 | 0.01 | 89.50% | 59.31 min |

Setting #1 exceeded the assignment target of 90% accuracy.

## Implementation
The submitted ResNet implementation uses residual blocks with Batch Normalization and ReLU, followed by global 8×8 average pooling and a fully connected classifier. The training script uses random cropping, horizontal flipping, CIFAR-10 normalization, SGD with momentum, weight decay, and a step-based learning-rate schedule.

## Source Note
The submitted evaluation spreadsheet describes Xavier initialization and no data augmentation, while the submitted Python implementation uses Kaiming initialization and applies random crop / horizontal flip augmentation. This repository preserves the behavior of the submitted Python implementation. The accuracy and training-time values above are the results reported in the assignment spreadsheet.

## Project Structure
```text
cifar10-resnet-classification/
├── config/
│   └── resnet_cifar10.yaml
├── models/
│   ├── __init__.py
│   └── resnet.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage
```bash
pip install -r requirements.txt
python -m scripts.train --config config/resnet_cifar10.yaml
python -m scripts.evaluate --config config/resnet_cifar10.yaml --checkpoint runs/resnet_cifar10/checkpoints/best.pth
python -m scripts.predict --config config/resnet_cifar10.yaml --checkpoint runs/resnet_cifar10/checkpoints/best.pth --image path/to/image.jpg
```
