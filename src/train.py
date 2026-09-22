import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASS_NAMES, HAM10000Dataset, load_metadata
from model import build_model
from transforms import get_train_transforms, get_eval_transforms


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # did not improve


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(data_dir, image_dirs, output_dir, epochs=30, batch_size=32, lr=1e-3, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = load_metadata(os.path.join(data_dir, "train.csv"))
    val_df = load_metadata(os.path.join(data_dir, "val.csv"))

    train_ds = HAM10000Dataset(train_df, image_dirs, transform=get_train_transforms())
    val_ds = HAM10000Dataset(val_df, image_dirs, transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = train_df["dx"].nunique()
    model = build_model(num_classes=num_classes, freeze_backbone=True).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["dx"].map({name: idx for idx, name in enumerate(CLASS_NAMES)}),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    early_stopping = EarlyStopping(patience=patience)
    best_state = None
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        improved = early_stopping.step(val_loss)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(output_dir, "best_model.pth"))
            print("  -> val_loss improved, model saved.")

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--image_dirs", nargs="+", default=[
        "data/HAM10000_images_part_1",
        "data/HAM10000_images_part_2",
    ])
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(
        args.data_dir, args.image_dirs, args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
    )
