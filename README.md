# CIFAR-10 Image Classification — Custom ResNet Architecture

> Custom ResNet-inspired CNN achieving **92.22% test accuracy** on CIFAR-10 in just **24 epochs** with only **1.67% overfitting gap**. Built with PyTorch, Albumentations, and One Cycle Policy scheduling.

## 🌐 Live Demo
**[View Live Demo →](https://meghana0824.github.io/Cifar10-Custom-ResNet/)**

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | 92.22% (Epoch 23) |
| **Final Training Accuracy** | 93.89% (Epoch 23) |
| **Final Test Loss** | 0.0004 |
| **Total Epochs** | 24 |
| **Training Time** | ~23 seconds per epoch |
| **Overfitting Gap** | Only 1.67% |

---

## 🎯 Training Progression

| Epoch | Learning Rate | Train Accuracy | Test Accuracy | Test Loss |
|-------|--------------|----------------|---------------|-----------|
| 0 | 0.004006 | 40.78% | 52.10% | 0.0026 |
| 5 | 0.038148 | 75.51% | 77.53% | 0.0013 |
| 10 | 0.028664 | 83.65% | 84.36% | 0.0009 |
| 15 | 0.019180 | 87.43% | 87.71% | 0.0007 |
| 20 | 0.009696 | 91.17% | 89.19% | 0.0006 |
| **23** | **0.004006** | **93.89%** | **92.22%** | **0.0004** |

---

## 📈 Class-wise Performance

| Class | Accuracy | Notes |
|-------|----------|-------|
| ✈ Plane | **100%** | Perfect classification |
| 🚗 Car | **100%** | Perfect classification |
| 🐦 Bird | **100%** | Perfect classification |
| 🐱 Cat | **100%** | Perfect classification |
| 🦌 Deer | **100%** | Perfect classification |
| 🐶 Dog | **87%** | Lower due to visual similarity with Cat |
| 🐸 Frog | **100%** | Perfect classification |
| 🐴 Horse | **100%** | Perfect classification |
| 🚢 Ship | **100%** | Perfect classification |
| 🚛 Truck | **83%** | Lower due to visual similarity with Car |

> **8 out of 10 classes achieved 100% accuracy**. Only Dog (87%) and Truck (83%) showed lower performance due to visual similarity with Cat and Car respectively.

---

## 🏗️ Model Architecture

Designed custom ResNet-inspired architecture (`models/A11.py`):

```
Input (32x32x3)
      │
PrepLayer: Conv2d(3→64, 3x3) → BatchNorm → ReLU          [RF: 3x3]
      │
Layer 1:  Conv2d(64→128, 3x3) → MaxPool → BatchNorm → ReLU  [RF: 11x11]
          + ResBlock 1 (skip connection)
      │
Layer 2:  Conv2d(128→256, 3x3) → MaxPool → BatchNorm → ReLU [RF: 27x27]
      │
Layer 3:  Conv2d(256→512, 3x3) → MaxPool → BatchNorm → ReLU [RF: 59x59]
          + ResBlock 2 (skip connection)
      │
MaxPool(4) → Fully Connected (512→10) → Log SoftMax        [RF: 88x88]
```

### Receptive Field Growth

| Layer | Receptive Field | Channels |
|-------|----------------|----------|
| PrepLayer | 3×3 | 64 |
| Layer 1 + ResBlock | 11×11 | 128 |
| Layer 2 | 27×27 | 256 |
| Layer 3 + ResBlock | 59×59 | 512 |
| Final MaxPool | 88×88 | 512 |

> Receptive field grows from 3×3 to 88×88 — larger than the 32×32 input, ensuring the model sees the full image with context.

---

## 🔄 Data Augmentation Pipeline (Albumentations)

```python
train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REFLECT),
    A.RandomCrop(height=32, width=32),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16,
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=CIFAR10_MEAN, p=0.5),
    ToTensorV2()
])
```

| Technique | Purpose |
|-----------|---------|
| **PadIfNeeded (40×40)** | Reflection padding before crop |
| **RandomCrop (32×32)** | Teaches position invariance |
| **HorizontalFlip (p=0.5)** | Doubles effective dataset size |
| **Normalize** | ImageNet statistics normalization |
| **CutOut (16×16)** | Forces robust feature learning |

---

## ⚙️ Training Configuration

### One Cycle Policy — Super Convergence

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.04006,          # Peak LR at epoch 5
    epochs=24,
    steps_per_epoch=1,
    pct_start=0.208,         # 20.8% warmup phase
    anneal_strategy="linear",
    cycle_momentum=False,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=10.0,
    final_div_factor=1.0
)
```

### SGD Optimizer

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.005
)
```

---

## 🔬 Learning Rate Range Test

LR range test conducted from 0.0001 to 0.01 over 10 epochs to identify optimal LR:

| Epoch | Learning Rate | Train Accuracy | Loss |
|-------|--------------|----------------|------|
| 1 | 0.0001 | 21.20% | 1.965 |
| 2 | 0.00109 | 36.26% | 1.570 |
| 3 | 0.00208 | 40.11% | 1.339 |
| 4 | 0.00307 | 40.94% | 1.294 |
| **5** | **0.00406** | **41.04%** | **1.372** |
| 6 | 0.00505 | 40.68% | 1.276 |
| 7 | 0.00604 | 35.17% | 1.359 |
| 8 | 0.00703 | 24.54% | 1.686 |

> **Finding:** Optimal learning rate identified at ~0.00406, informing the One Cycle Policy max LR of 0.04006.

---

## 💡 Key Techniques & Impact

| Technique | Impact |
|-----------|--------|
| **One Cycle Policy** | Rapid convergence from 52% → 92% in 24 epochs |
| **Residual Connections** | Enables deep network without vanishing gradients |
| **CutOut Augmentation** | Reduces overfitting gap to only 1.67% |
| **Batch Normalization** | Stabilizes training, enables higher LR |
| **Weight Decay (0.005)** | L2 regularization prevents overfitting |
| **SGD + Momentum (0.9)** | Better generalization than Adam |

---

## 🗂️ Project Structure

```
Cifar10-Custom-ResNet/
├── model/
│   ├── resnet.py              # Custom ResNet architecture (A11)
│   └── blocks.py              # Residual block definitions
├── data/
│   └── augmentation.py        # Albumentations pipeline
├── docs/
│   └── index.html             # Live demo page
├── train.py                   # Training loop with One Cycle Policy
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision albumentations tqdm matplotlib numpy opencv-python
```

### Quick Start

```python
from model.resnet import get_model
from data.augmentation import get_train_transforms, get_test_transforms
import torch.optim as optim
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=10).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.04006,
    epochs=24,
    steps_per_epoch=len(train_loader),
    pct_start=0.208,
    anneal_strategy="linear",
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=10.0
)

loss_func = nn.CrossEntropyLoss()

# Train
python train.py
```

---

## 🛠️ Tools & Frameworks

| Tool | Version | Purpose |
|------|---------|---------|
| **PyTorch** | ≥2.0.0 | Neural network & training |
| **TorchVision** | ≥0.15.0 | CIFAR-10 dataset loading |
| **Albumentations** | ≥1.3.0 | Advanced data augmentation |
| **NumPy** | ≥1.24.0 | Array operations |
| **Matplotlib** | ≥3.7.0 | Visualizations |
| **tqdm** | ≥4.65.0 | Progress tracking |
| **OpenCV** | ≥4.7.0 | Image processing backend |

---

## 🏆 Key Achievements

- ✅ **92.22% test accuracy** achieved in just 24 epochs
- ✅ **8 out of 10 classes** with 100% accuracy
- ✅ **Efficient training** — ~23 seconds per epoch
- ✅ **Minimal overfitting** — only 1.67% train/test gap
- ✅ **Class-wise analysis** — full misclassification tracking
- ✅ **Modern techniques** — One Cycle Policy, Albumentations, residual connections

---

## 🎓 Skills Demonstrated

- **Deep Learning** — Custom architecture design, optimization, hyperparameter tuning
- **Computer Vision** — Image classification, augmentation, feature extraction
- **MLOps** — Training pipeline, model evaluation, performance tracking
- **Software Engineering** — Modular design, clean code, reusable components
- **Data Science** — Statistical analysis, visualization, result interpretation

---

## 📝 Notes

- Model uses `log_softmax` in forward pass, compatible with `CrossEntropyLoss`
- Batch size 512 optimized for GPU memory and training speed
- Training designed for Google Colab but works on any PyTorch environment
- Model parameters: ~6.5M (approximate)

---

## 👩‍💻 Author

**Lakshmimeghana Uppalapati**
- 📧 Email: [ulakshmi081234@gmail.com](mailto:ulakshmi081234@gmail.com)
- 💻 GitHub: [github.com/meghana0824](https://github.com/meghana0824)
- 🌐 Live Demo: [meghana0824.github.io/Cifar10-Custom-ResNet](https://meghana0824.github.io/Cifar10-Custom-ResNet/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
