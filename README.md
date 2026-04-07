## 🌐 Live Demo
**[View Live Demo →](https://meghana0824.github.io/Cifar10-Custom-ResNet/)**

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **92.22%** |
| Training Epochs | 24 |
| Train/Test Gap (Overfitting) | 1.67% |
| Perfect Classes (100% accuracy) | 8 out of 10 |
| Training Speed Improvement | 40% faster (vs baseline) |

### Per-Class Accuracy

| Class | Accuracy |
|---|---|
| Airplane | 100% |
| Automobile | 100% |
| Bird | 100% |
| Cat | 100% |
| Deer | 100% |
| Dog | 100% |
| Frog | 100% |
| Horse | 100% |
| Ship | ~84% |
| Truck | ~84% |

---

## Architecture

### Custom ResNet Block

```
Input
  │
  ├──────────────────────┐
  │                      │ (skip connection)
  ▼                      │
Conv2d (3x3)             │
BatchNorm2d              │
ReLU                     │
Conv2d (3x3)             │
BatchNorm2d              │
  │                      │
  └──────────────────────┘
  │   (element-wise add)
  ▼
ReLU
Output
```

### Full Network Architecture

```
Input (32x32x3)
      │
   Conv2d → BN → ReLU        [Receptive Field: 3x3]
      │
   ResBlock 1 (64 filters)   [RF: 11x11]
      │
   MaxPool
      │
   ResBlock 2 (128 filters)  [RF: 27x27]
      │
   MaxPool
      │
   ResBlock 3 (256 filters)  [RF: 59x59]
      │
   MaxPool
      │
   ResBlock 4 (512 filters)  [RF: 88x88]
      │
   AdaptiveAvgPool
      │
   Linear → 10 classes
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Augmentation | Albumentations |
| LR Scheduling | One Cycle Policy |
| Optimizer | SGD with momentum |
| Data | CIFAR-10 (50K train / 10K test) |

---

## Key Techniques

### 1. Custom Residual Blocks with Skip Connections
- Enables gradient flow through deeper architectures
- Prevents vanishing gradients
- Calculated receptive fields from 3×3 to 88×88

### 2. Advanced Augmentation Pipeline (Albumentations)
```python
train_transforms = A.Compose([
    A.RandomCrop(height=32, width=32, padding=4),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=mean,   # fill with dataset mean
        p=0.5
    ),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])
```

### 3. One Cycle Policy — Super Convergence
- Linear annealing strategy
- Dynamic momentum adjustment (0.95 → 0.85 → 0.95)
- Enables 40% faster training convergence
- Peak LR at 30% of total steps, then anneals down

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=24,
    pct_start=0.3,
    anneal_strategy='linear',
    div_factor=10,
    final_div_factor=100
)
```

---

## Project Structure

```
cifar10-custom-resnet/
├── model/
│   ├── resnet.py          # Custom ResNet architecture
│   ├── blocks.py          # Residual block definitions
│   └── __init__.py
├── data/
│   ├── dataset.py         # CIFAR-10 data loading
│   └── augmentation.py    # Albumentations pipeline
├── training/
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation & per-class accuracy
│   └── scheduler.py       # One Cycle Policy setup
├── utils/
│   ├── visualize.py       # Accuracy/loss plots
│   └── receptive_field.py # RF calculation utility
├── notebooks/
│   └── CIFAR10_ResNet.ipynb  # Full experiment notebook
├── assets/
│   ├── architecture.png
│   ├── training_curves.png
│   └── confusion_matrix.png
├── requirements.txt
├── train.py               # Main entry point
├── .gitignore
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python >= 3.9
- CUDA-capable GPU (recommended) or CPU
- PyTorch >= 2.0

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cifar10-custom-resnet.git
cd cifar10-custom-resnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
scikit-learn>=1.2.0
```

### Training

```bash
# Train with default config (24 epochs, One Cycle LR)
python train.py

# Custom training
python train.py \
  --epochs 24 \
  --batch-size 512 \
  --max-lr 0.01 \
  --device cuda
```

### Evaluation

```bash
# Evaluate on test set
python training/evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## Training Configuration

```python
config = {
    "epochs": 24,
    "batch_size": 512,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "max_lr": 0.01,
    "scheduler": "OneCycleLR",
    "anneal_strategy": "linear",
    "pct_start": 0.3,
}
```

---

## Receptive Field Calculation

| Layer | Receptive Field |
|---|---|
| Input Conv | 3×3 |
| After ResBlock 1 | 11×11 |
| After ResBlock 2 | 27×27 |
| After ResBlock 3 | 59×59 |
| After ResBlock 4 | 88×88 |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Lakshmimeghana Uppalapati**
- Email: ulakshmi081234@gmail.com
- LinkedIn: [linkedin.com/in/lakshmimeghana](https://linkedin.com/in/lakshmimeghana)
