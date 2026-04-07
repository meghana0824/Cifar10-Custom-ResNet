import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model.resnet import get_model
from data.augmentation import get_train_transforms, get_test_transforms


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


def get_dataloaders(batch_size=512):
    from torchvision import datasets as dsets
    train_data = dsets.CIFAR10(root='./data', train=True,  download=True)
    test_data  = dsets.CIFAR10(root='./data', train=False, download=True)

    train_dataset = CIFAR10Dataset(train_data, transform=get_train_transforms())
    test_dataset  = CIFAR10Dataset(test_data,  transform=get_test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    train_loader, test_loader = get_dataloaders(config['batch_size'])
    model = get_model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['max_lr'] / 10,
        momentum=0.9,
        weight_decay=5e-4
    )

    # One Cycle Policy — enables super-convergence
    # Linear annealing with dynamic momentum adjustment
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['max_lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.3,
        anneal_strategy='linear',
        div_factor=10,
        final_div_factor=100
    )

    best_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        train_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f"{train_loss/total:.4f}",
                'Acc':  f"{100.*correct/total:.2f}%",
                'LR':   f"{scheduler.get_last_lr()[0]:.6f}"
            })

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} | Train Acc: {100.*correct/total:.2f}% | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"  → New best model saved! ({best_acc:.2f}%)")

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)

    config = {
        'epochs':     24,
        'batch_size': 512,
        'max_lr':     0.01,
    }
    train(config)
