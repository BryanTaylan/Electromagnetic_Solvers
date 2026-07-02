import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import time

DATASET_DIR = Path("dataset_224")
BATCH_SIZE = 16
LR_PROBE = 1e-3      
LR_FINETUNE = 1e-4   
EPOCHS = 50
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FieldDataset224(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = np.load(row["filepath"]).astype(np.float32)
        x = torch.tensor(sample)
        if self.transform:
            x = self.transform(x)
        y = int(row["class_label"])
        return x,y
    
def train_linear_probe(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_PROBE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0 
    patience = 10
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == y ).sum().item()
            train_total += y.size(0)  # ← add this line

        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_resnet_probe.pt")
            print(f"  → New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs")
                break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")
    return history

def train_finetune(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0 
    patience = 10
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == y ).sum().item()
            train_total += y.size(0)  # ← add this line

        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_resnet_finetune.pt")
            print(f"  → New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs")
                break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")
    return history

def evaluate(model, test_loader, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extends(y.cpu().numpy())
    
    inference_time = (time.time() - start_time) / len(all_preds)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    cm = np.zeros((4,4), dtype=int)
    for true, pred in zip (all_labels, all_preds):
        cm[true][pred] += 1
    
    per_class_acc = cm.diagonal() / cm.sum(axis = 1)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Per-class Accuracy: {per_class_acc}")
    print(f"Inference time per sample: {inference_time*1000:.2f}ms")
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, cm, per_class_acc, inference_time


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # load splits
    train_df = pd.read_csv(DATASET_DIR / "train_split.csv")
    val_df   = pd.read_csv(DATASET_DIR / "val_split.csv")
    test_df  = pd.read_csv(DATASET_DIR / "test_split.csv")

    # get ResNet18 pretrained weights and transform
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    # create datasets
    train_dataset = FieldDataset224(train_df, transform=transform)
    val_dataset   = FieldDataset224(val_df,   transform=transform)
    test_dataset  = FieldDataset224(test_df,  transform=transform)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Linear Probe ──
    print("\n=== ResNet18 Linear Probe ===")
    model = resnet18(weights=weights).to(device)
    for param in model.parameters():
        param.requires_grad = False          # freeze all layers
    model.fc = nn.Linear(512, 4).to(device) # replace final layer
    
    history_probe = train_linear_probe(model, train_loader, val_loader)
    acc_probe, cm_probe, per_class_probe, inf_time_probe = evaluate(model, test_loader, "best_resnet_probe.pt")

    # ── Fine-tune ──
    print("\n=== ResNet18 Fine-tune ===")
    model = resnet18(weights=weights).to(device)
    model.fc = nn.Linear(512, 4).to(device)
    for param in model.parameters():
        param.requires_grad = True           # unfreeze all layers

    history_ft = train_finetune(model, train_loader, val_loader)
    acc_ft, cm_ft, per_class_ft, inf_time_ft = evaluate(model, test_loader, "best_resnet_finetune.pt")

    # ── Plot curves ──
    for history, name in [(history_probe, "probe"), (history_ft, "finetune")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"],   label="Val Loss")
        axes[0].set_title(f"Loss — ResNet18 {name}")
        axes[0].legend()
        axes[1].plot(history["train_acc"], label="Train Acc")
        axes[1].plot(history["val_acc"],   label="Val Acc")
        axes[1].set_title(f"Accuracy — ResNet18 {name}")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"resnet_{name}_curves.png", dpi=150)
        plt.close()
        print(f"Saved resnet_{name}_curves.png")

if __name__ == "__main__":
    main()       