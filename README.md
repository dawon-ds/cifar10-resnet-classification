# CIFAR-10 Image Classification with ResNet

[Portfolio Page — ResNet & Vision Transformer](https://app.notion.com/p/3e968564df5a81de83eded127aa1047f)

Implemented a ResNet-32 classifier from scratch for CIFAR-10 and examined classification accuracy and training time under two learning-rate settings.

## Approach

**Technologies:** Python, PyTorch, Torchvision, NumPy, PyYAML.

### 1. Data preparation

CIFAR-10 contains 50,000 training images and a separate 10,000-image test partition. The training script shuffles the official training partition with a fixed seed (**54321**) and divides it into **45,000 training** and **5,000 validation** images.

- **Training augmentation:** Apply `RandomCrop(32, padding=4)` and `RandomHorizontalFlip` before converting images to tensors and normalizing with CIFAR-10 channel statistics.
- **Validation processing:** Use a separate dataset instance with deterministic tensor conversion and normalization, without random augmentation.
- **Test processing:** The evaluation script uses the official test partition with the same deterministic normalization.

### 2. ResNet-32 architecture

The model is implemented in `models/resnet.py`, rather than loaded from pretrained weights. An initial 3×3 convolution maps RGB input to 16 channels, followed by **three stages** with **five residual blocks per stage** and channel dimensions **16 → 32 → 64**.

Each residual block contains two 3×3 convolutions with Batch Normalization. The output of the convolutional path is added to a shortcut path before the final ReLU activation. When a new stage changes channel count and spatial resolution, `IdentityPadding` pads the shortcut's channels and downsamples it using a stride-2, 1×1 max-pooling operation. An 8×8 average-pooling layer and a fully connected layer output logits for the **10 CIFAR-10 classes**.

Convolutional weights use **Kaiming normal initialization**; Batch Normalization scale and bias parameters are initialized to one and zero, respectively.

### 3. Training configuration and evaluation

The training loop minimizes **cross-entropy loss** using SGD with **momentum 0.9**, **batch size 128**, and **weight decay 1e-4**. The configuration specifies up to **200 epochs**.

The `MultiStepLR` scheduler is advanced **after every optimizer step**, with milestones at **32,000** and **48,000 training steps** and a learning-rate multiplier of **0.1** at each milestone. These milestones are measured in optimizer updates, not epochs.

- **Setting #1:** Initial learning rate **0.1** (the supplied YAML configuration).
- **Setting #2:** Initial learning rate **0.01**, with the remaining documented training settings unchanged.

After each epoch, the script computes validation accuracy from the predicted class (`argmax`) and ground-truth label. When validation accuracy improves, it saves `runs/resnet_cifar10/checkpoints/best.pth`. The separate evaluation script measures accuracy on the official CIFAR-10 test partition, while the prediction script classifies an individual image using a saved checkpoint.

## Results

| Setting | Learning Rate | Reported Accuracy | Training Time |
|---|---:|---:|---:|
| #1 | 0.1 | 92.27% | 60.44 min |
| #2 | 0.01 | 89.50% | 59.31 min |

In the submitted assignment results, Setting #1 recorded **2.77 percentage points** higher accuracy than Setting #2, with similar training times. It also exceeded the assignment's 90% accuracy target. The recorded difference describes these two experimental settings; it does not establish that 0.1 is generally the optimal learning rate.

The values are taken from the **submitted evaluation spreadsheet** and are not verified reproduction results from the code currently in this repository; see [Limitations](#limitations).

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Train with the supplied configuration (Setting #1, learning rate 0.1):

```bash
python -m scripts.train --config config/resnet_cifar10.yaml
```

To run Setting #2, set `training.learning_rate` to `0.01` in a copy of the configuration and choose a different `output.dir` so checkpoints from the two runs do not overwrite each other.

Evaluate a saved checkpoint on the official CIFAR-10 test partition:

```bash
python -m scripts.evaluate --config config/resnet_cifar10.yaml --checkpoint runs/resnet_cifar10/checkpoints/best.pth
```

Predict the class of a single image:

```bash
python -m scripts.predict --config config/resnet_cifar10.yaml --checkpoint runs/resnet_cifar10/checkpoints/best.pth --image path/to/image.jpg
```

The checkpoint path is created by a completed training run. Torchvision downloads CIFAR-10 when it is not already cached.

## Limitations

The submitted evaluation spreadsheet lists **Xavier initialization** and **no data augmentation**, whereas the submitted Python implementation uses **Kaiming initialization**, random cropping, and horizontal flipping. The repository documents and retains the Python implementation as written; the accuracy and training-time figures above remain the historical values in the assignment report, **not independently reproduced measurements** from this code.

The reported accuracy's evaluation split is not independently established here. The current scripts select checkpoints by validation accuracy and provide a separate test evaluation, so a rigorous new comparison would require rerunning both settings under a documented, matching evaluation protocol and recording its outputs.

## Data and Third-party Resources

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), downloaded through Torchvision; raw images are not committed to this repository.
- **Implementation:** The ResNet-32 architecture, training, evaluation, and image-prediction scripts are included in the repository.

Dependencies are listed in `requirements.txt` without pinned versions, so the file does not define an exact historical software environment. Follow the source dataset's applicable terms when using or redistributing third-party material.
