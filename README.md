# CIFAR-10 Image Classification with ResNet

[Portfolio Page](https://app.notion.com/p/3e968564df5a8143bc80da0db7f081d4)

Implemented a ResNet-32 classifier for CIFAR-10 and compared two learning-rate settings.


## Project Overview

- **Task:** CIFAR-10 image classification
- **Model:** ResNet-32
- **Classes:** 10
- **Batch size:** 128
- **Optimizer:** SGD with Momentum
- **Weight decay:** 1e-4
- **Maximum epochs:** 200
- **Learning-rate schedule:** MultiStepLR at steps 32,000 and 48,000

## Data

### Data & Training Workflow

The implementation splits the official CIFAR-10 training partition into **45,000 training** and **5,000 validation** images. Training uses random crop and horizontal flip; validation uses a separate dataset instance with deterministic normalization.

The model is trained with cross-entropy loss and SGD with momentum **0.9**. The learning-rate scheduler advances **once per optimizer update**, so the 32,000 and 48,000 milestones represent training steps rather than epochs.

The best checkpoint is selected by validation accuracy and saved to `runs/resnet_cifar10/checkpoints/best.pth`. The separate evaluation script uses the official test partition. Torchvision downloads CIFAR-10 automatically.

The default configuration uses learning rate 0.1. For Setting #2, change it to 0.01 and use a separate output directory to preserve both runs.

## Approach

The submitted ResNet implementation uses residual blocks with Batch Normalization and ReLU, followed by global 8×8 average pooling and a fully connected classifier. The training script uses random cropping, horizontal flipping, CIFAR-10 normalization, SGD with momentum, weight decay, and a step-based learning-rate schedule.

## Results

| Setting | Learning Rate | Accuracy | Training Time |
|---|---:|---:|---:|
| #1 | 0.1 | 92.27% | 60.44 min |
| #2 | 0.01 | 89.50% | 59.31 min |

Setting #1 exceeded the assignment target of 90% accuracy **in the submitted report**. The report and available code differ in initialization and augmentation, so these values are not verified reproduction results of the present code.

### Interpretation

Setting #1 has a reported accuracy **2.77 percentage points** higher than Setting #2, with similar recorded training times.

## Repository Structure

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

## Limitations

### Experiment Notes

The submitted evaluation spreadsheet describes Xavier initialization and no data augmentation, while the submitted Python implementation uses Kaiming initialization and applies random crop / horizontal flip augmentation. This repository preserves the behavior of the submitted Python implementation. The accuracy and training-time values above are the results reported in the assignment spreadsheet.

The report/code discrepancy described above prevents treating the table as a verified benchmark of the present code. Repeated seeds and a clearly documented evaluation split would make a future comparison stronger.

## Review

Implementing residual blocks provided practice with shortcut connections, normalization, initialization, augmentation, and step-based scheduling. The repository separates training, evaluation, and single-image prediction so each stage can be inspected independently.

## Data and Third-party Resources

- **Data:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), downloaded through Torchvision; raw images are not committed.
- **Implementation:** the ResNet-32 training code is included. Report-versus-code differences are documented in Limitations.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
