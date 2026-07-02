import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DATASET_DIR = Path("dataset")
METADATA_CSV = DATASET_DIR / "metadata.csv"
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 50
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FieldDataset(Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sample = np.load(row["filepath"]).astype(np.float32)

        for c in range(sample.shape[0]):
            max_val = np.abs(sample[c]).max()
            if max_val > 0:
                sample[c] = sample[c] / max_val

        x = torch.tensor(sample, dtype=torch.float32)
        y = int(row["class_label"])
        return x, y

def split_by_frequency(df,train_frac=0.70, val_frac = 0.15):
    train_dfs, val_dfs, test_dfs = [], [], []

    for label in df["class_label"].unique():
        class_df = df[df["class_label"] == label].sort_values("frequency").reset_index(drop=True)
        n = len(class_df)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_dfs.append(class_df.iloc[:n_train])
        val_dfs.append(class_df.iloc[n_train:n_train + n_val])
        test_dfs.append(class_df.iloc[n_train + n_val:])

    train_df = pd.concat(train_dfs).reset_index(drop = True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop = True)

    return train_df, val_df, test_df

class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    patience = 10
    epochs_without_improvement = 0
    
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
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
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
            torch.save(model.state_dict(), "best_cnn.pt")
            print(f"  → New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs")
                break

    
    return history

def evaluate(model, test_loader):
    model.load_state_dict(torch.load("best_cnn.pt", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

    num_classes = 4
    cm = np.zeros((num_classes, num_classes), dtype = int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1
    
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, cm

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    df = pd.read_csv(METADATA_CSV)

    train_df, val_df, test_df = split_by_frequency(df)

    train_df.to_csv(DATASET_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATASET_DIR / "val_split.csv", index=False)
    test_df.to_csv(DATASET_DIR / "test_split.csv", index=False)
    print("Splits saved!")

    train_dataset = FieldDataset(train_df)
    val_dataset = FieldDataset(val_df)
    test_dataset = FieldDataset(test_df)

    train_loader = DataLoader(train_dataset ,batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset ,batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset ,batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=4).to(device)

    history = train_model(model, train_loader, val_loader)

    accuracy , cm = evaluate(model, test_loader)

    # 8. plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy curves
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    plt.show()
    print("Saved cnn_training_curves.png")


if __name__ == "__main__":
    main()



