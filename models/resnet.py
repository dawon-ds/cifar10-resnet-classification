import torch.nn as nn
import torch.nn.functional as F


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        x = F.pad(x, [0, 0, 0, 0, 0, self.add_channels])
        return self.pooling(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down_sample = (
            IdentityPadding(in_channels, out_channels, stride)
            if down_sample else None
        )

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.down_sample is not None:
            shortcut = self.down_sample(shortcut)

        return self.relu(x + shortcut)


class ResNet32(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_layers = 5
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layers_2n = self._make_layer(16, 16, stride=1)
        self.layers_4n = self._make_layer(16, 32, stride=2)
        self.layers_6n = self._make_layer(32, 64, stride=2)

        self.pool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride):
        blocks = [
            ResidualBlock(
                in_channels, out_channels, stride,
                down_sample=(stride == 2)
            )
        ]
        for _ in range(self.num_layers - 1):
            blocks.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_model(num_classes=10):
    return ResNet32(num_classes=num_classes)
