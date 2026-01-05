# =========================
# ⚡ Setup & Imports
# =========================
import os
import json
import glob
import time
import random
import multiprocessing as mp
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from PIL import Image
import cv2

# Set OpenCV thread count
cv2.setNumThreads(0)

# -------- Reproducibility


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 📂 Paths & Configuration
# =========================
DATASET_ROOT = r"D:\Python\Projects\Brain Stroke CT\ozguraslank\brain-stroke-ct-dataset\versions\3\Brain_Stroke_CT_Dataset"
WORK_DIR = "./ct_memmap_cache"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Bleeding", "Ischemia"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES)}
BATCH_SIZE = 16
NUM_WORKERS = 0  # Set to 0 for Windows stability, or higher for Linux/Mac
EPOCHS = 10

# =========================
# 📖 Dataset Classes
# =========================


class CTMemmapDataset(Dataset):
    def __init__(self, imgs_path, labels_path, augment=False):
        self.imgs = np.load(imgs_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.augment = augment

        aug_list = []
        if augment:
            aug_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]

        self.tf = transforms.Compose(aug_list + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_np = self.imgs[idx]
        label = int(self.labels[idx])
        img_pil = Image.fromarray(img_np)
        img_tensor = self.tf(img_pil)
        return img_tensor, label


class IndexedCTDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]

# =========================
# 🧰 Helper Functions
# =========================


def scan_pngs(root, class_to_idx):
    files, labels = [], []
    for cls in class_to_idx:
        pattern = os.path.join(root, cls, "PNG", "*.png")
        paths = sorted(glob.glob(pattern))
        if len(paths) == 0:
            raise RuntimeError(
                f"No images found for class '{cls}'. Check paths.")
        files.extend(paths)
        labels.extend([class_to_idx[cls]] * len(paths))
    return files, np.array(labels, dtype=np.int64)


def run_epoch(model, loader, device, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for xb, yb in loop:
        xb, yb = xb.to(device, non_blocking=True), yb.to(
            device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# =========================
# 🎯 Main Execution
# =========================
if __name__ == "__main__":
    # 1. Setup
    mp.set_start_method("spawn", force=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # 2. Data Scanning
    all_files, all_labels = scan_pngs(DATASET_ROOT, CLASS_TO_IDX)
    N = len(all_files)
    print(f"Dataset found. Total images: {N}")

    # 3. Memmap Cache Creation
    imgs_npy = os.path.join(
        WORK_DIR, f"images_{IMG_SIZE[0]}x{IMG_SIZE[1]}_uint8.npy")
    labels_npy = os.path.join(WORK_DIR, "labels_int64.npy")

    if not os.path.exists(imgs_npy):
        print("Preprocessing images to memmap cache...")
        imgs_mm = np.lib.format.open_memmap(
            imgs_npy, mode="w+", dtype=np.uint8, shape=(N, IMG_SIZE[0], IMG_SIZE[1], 3))
        labels_mm = np.lib.format.open_memmap(
            labels_npy, mode="w+", dtype=np.int64, shape=(N,))

        for i, (p, lbl) in enumerate(zip(all_files, all_labels)):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
            imgs_mm[i], labels_mm[i] = img, lbl
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{N}")
    else:
        print("Using existing memmap cache.")

    # 4. Splits & Loaders
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.arange(N), all_labels))

    base_train = CTMemmapDataset(imgs_npy, labels_npy, augment=True)
    base_val = CTMemmapDataset(imgs_npy, labels_npy, augment=False)

    train_loader = DataLoader(IndexedCTDataset(base_train, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(IndexedCTDataset(base_val, val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 5. Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    # 6. Training Loop
    best_val_acc = 0.0
    best_model_path = os.path.join(WORK_DIR, "best_resnet18.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, device, criterion, optimizer, train=True)
        val_loss, val_acc, val_p, val_l = run_epoch(
            model, val_loader, device, criterion, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" ↳ New best model saved!")

    # 7. Final Evaluation
    print("\nFinal Evaluation Report:")
    model.load_state_dict(torch.load(best_model_path))
    _, _, final_preds, final_labels = run_epoch(
        model, val_loader, device, criterion, train=False)
    print(classification_report(final_labels,
          final_preds, target_names=CLASS_NAMES))

    # 8. Confusion Matrix Plot
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
