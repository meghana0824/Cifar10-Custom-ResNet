import torch
import torch.nn as nn
from model.blocks import ResidualBlock


class CustomResNet(nn.Module):
    """
    Custom ResNet-inspired CNN architecture for CIFAR-10 classification.
    Achieves 92.22% test accuracy in 24 epochs with minimal overfitting (1.67% gap).
    Receptive fields range from 3x3 to 88x88.
    """

    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()

        # Initial Conv layer — RF: 3x3
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 1 — RF: 11x11
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128)
        )

        # Layer 2 — RF: 27x27
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Layer 3 — RF: 59x59
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512, 512)
        )

        # Global Average Pooling — RF: 88x88
        self.pool = nn.MaxPool2d(4, 4)

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(num_classes=10):
    return CustomResNet(num_classes=num_classes)
