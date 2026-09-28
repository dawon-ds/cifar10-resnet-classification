import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from models import build_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/resnet_cifar10.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    test_data = torchvision.datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )
    loader = DataLoader(
        test_data,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    model = build_model(cfg["model"]["num_classes"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / len(test_data)
    print(f"test_accuracy={accuracy:.2f}%")


if __name__ == "__main__":
    main()
