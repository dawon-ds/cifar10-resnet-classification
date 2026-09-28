import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import build_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/resnet_cifar10.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root="data", train=True, transform=train_transform, download=True
    )
    val_data = torchvision.datasets.CIFAR10(
        root="data", train=True, transform=eval_transform, download=True
    )

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    val_size = cfg["data"]["val_size"]
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        train_data,
        batch_size=cfg["training"]["batch_size"],
        sampler=SubsetRandomSampler(train_indices),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["training"]["batch_size"],
        sampler=SubsetRandomSampler(val_indices),
    )

    model = build_model(cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["training"]["lr_milestones"],
        gamma=cfg["training"]["lr_gamma"],
    )

    checkpoint_dir = Path(cfg["output"]["dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()

        val_acc = 100.0 * correct / len(val_indices)
        print(
            f"epoch={epoch + 1} "
            f"val_accuracy={val_acc:.2f}% "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                },
                checkpoint_dir / "best.pth",
            )


if __name__ == "__main__":
    main()
