import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_train_transforms():
    """
    Advanced augmentation pipeline using Albumentations.
    Techniques: RandomCrop, HorizontalFlip, CutOut (CoarseDropout).
    Improves model generalization and robustness for production-grade classification.
    """
    return A.Compose([
        A.PadIfNeeded(min_height=40, min_width=40,
                      border_mode=0, value=0, p=1.0),
        A.RandomCrop(height=32, width=32, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=[v * 255 for v in CIFAR10_MEAN],
            p=0.5
        ),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])


def get_test_transforms():
    """Minimal transforms for test set — only normalize."""
    return A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])
