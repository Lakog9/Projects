"""
Drug-Drug Interaction Prediction: PyTorch MLP vs k-NN Baseline
Author: Petros Kogios
Dataset: DDI-Bench (DrugBank)
"""
import pickle
import json
import time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === PATHS ===
FEATS_PATH  = "DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl"
REL2ID_PATH = "DDI_Ben/TextDDI/data/drugbank_random/relation2id.json"
TRAIN_PATH  = "DDI_Ben/DDI_Ben/data/drugbank_random/train.txt"
TEST_PATH   = "DDI_Ben/DDI_Ben/data/drugbank_random/test_S0.txt"

# Load data
with open(FEATS_PATH, "rb") as f:
    feats_dict = pickle.load(f)
with open(REL2ID_PATH, "r") as f:
    relation_map = json.load(f)

node_ids = feats_dict["Node ID"]
morgan_all = feats_dict["Morgan_Features"]
rd2d_all = feats_dict["RDKit2D_Features"]
nid_to_idx = {int(node_ids[i]): i for i in range(len(node_ids))}

print(f"Drugs: {len(node_ids)}, Morgan dim: {len(morgan_all[0])}, RDKit2D dim: {len(rd2d_all[0])}")

def load_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                pairs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return pairs

def build_pair_features(pairs):
    X_list, y_list = [], []
    for a, b, label in pairs:
        idx_a, idx_b = nid_to_idx[a], nid_to_idx[b]
        feat = np.concatenate([
            np.asarray(morgan_all[idx_a], dtype=np.float32),
            np.asarray(rd2d_all[idx_a], dtype=np.float32),
            np.asarray(morgan_all[idx_b], dtype=np.float32),
            np.asarray(rd2d_all[idx_b], dtype=np.float32),
        ])
        X_list.append(feat)
        y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

print("Loading pairs...")
train_pairs = load_pairs(TRAIN_PATH)
test_pairs = load_pairs(TEST_PATH)
print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

print("Building feature matrices (~30s)...")
t0 = time.time()
X_train, y_train = build_pair_features(train_pairs)
X_test, y_test = build_pair_features(test_pairs)
print(f"Done in {time.time()-t0:.1f}s | X_train: {X_train.shape}, X_test: {X_test.shape}")

# Normalize
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std == 0] = 1.0
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

# DataLoaders
BATCH_SIZE = 256
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_norm), torch.from_numpy(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# Model
INPUT_DIM = X_train.shape[1]
NUM_CLASSES = len(set(y_train))

class DDI_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.network(x)

model = DDI_MLP(INPUT_DIM, NUM_CLASSES).to(device)
print(f"\nModel: {INPUT_DIM} -> 512 -> 256 -> {NUM_CLASSES}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
EPOCHS = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

train_losses, train_accs, test_accs = [], [], []

print(f"\nTraining for {EPOCHS} epochs...")
print("-" * 65)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)

    train_loss = epoch_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            correct += (model(X_batch).argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)
    test_acc = correct / total
    test_accs.append(test_acc)
    scheduler.step(train_loss)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

print("-" * 65)
print(f"Best test accuracy: {max(test_accs):.4f} (epoch {np.argmax(test_accs)+1})")

# Final evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

mlp_accuracy = accuracy_score(all_labels, all_preds)
mlp_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
print(f"\n=== FINAL RESULTS ===")
print(f"MLP Test Accuracy:  {mlp_accuracy:.4f}")
print(f"MLP F1 (weighted):  {mlp_f1:.4f}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(range(1, EPOCHS+1), train_losses, "b-o", markersize=4)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, EPOCHS+1), train_accs, "b-o", markersize=4, label="Train")
axes[1].plot(range(1, EPOCHS+1), test_accs, "r-s", markersize=4, label="Test")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Train vs Test Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ddi_mlp_results.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved: ddi_mlp_results.png")
