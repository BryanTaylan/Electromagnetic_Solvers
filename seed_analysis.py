import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import open_clip
import time

from train_cnn import FieldDataset, SimpleCNN, train_model, evaluate as evaluate_cnn, split_by_frequency
from train_resnet import FieldDataset224, train_linear_probe, train_finetune, evaluate as evaluate_resnet
from train_clip import CLIPClassifier, train_clip, evaluate as evaluate_clip

DATASET_DIR = Path("dataset")
DATASET_224_DIR = Path("dataset_224")
BATCH_SIZE = 16
LR = 1e-3
LR_FINETUNE = 1e-4
EPOCHS = 50
SEEDS = [0,1,2,3,4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    train_df = pd.read_csv(DATASET_DIR / "train_split.csv")
    val_df = pd.read_csv(DATASET_DIR / "val_split.csv")
    test_df = pd.read_csv(DATASET_DIR / "test_split.csv")

    train_df_224 = pd.read_csv(DATASET_224_DIR / "train_split.csv")
    val_df_224 = pd.read_csv(DATASET_224_DIR / "val_split.csv")
    test_df_224 = pd.read_csv(DATASET_224_DIR / "test_split.csv")

    results = {
        "cnn": [],
        "resnet_probe": [],
        "resnet_ft": [],
        "clip": [],
    }

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        set_seed(seed)

        print("\n--- Baseline CNN ---")
        train_dataset = FieldDataset(train_df)
        val_dataset = FieldDataset(val_df)
        test_dataset = FieldDataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = SimpleCNN(num_classes=4).to(device)
        train_model(model,train_loader,val_loader)
        acc,cm,per_class, inf_time = evaluate_cnn(model, test_loader)
        results["cnn"].append(acc)

        print("\n--- ResNet18 Linear Probe ---")
        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        train_dataset = FieldDataset224(train_df_224, transform=transform)
        val_dataset = FieldDataset224(val_df_224, transform=transform)
        test_dataset = FieldDataset224(test_df_224,transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = resnet18(weights=weights).to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512,4).to(device)
        train_linear_probe(model, train_loader, val_loader)
        acc, cm, per_class, inf_time = evaluate_resnet(model, test_loader, "best_resnet_probe.pt")
        results["resnet_probe"].append(acc)

        print("\n--- ResNet18 Fine-tune ---")
        model = resnet18(weights=weights).to(device)
        model.fc = nn.Linear(512,4).to(device)
        for param in model.parameters():
            param.requires_grad = True
        train_finetune(model, train_loader, val_loader)
        acc, cm, per_class, inf_time = evaluate_resnet(model, test_loader, "best_resnet_finetune.pt")
        results["resnet_ft"].append(acc)

        print("\n--- OpenCLIP ---")
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        classifier = CLIPClassifier(feature_dim=512, num_classes=4).to(device)
        train_dataset = FieldDataset224(train_df_224, transform=preprocess)
        val_dataset = FieldDataset224(val_df_224, transform=preprocess)
        test_dataset = FieldDataset224(test_df_224, transform=preprocess)
        train_loader = DataLoader(train_dataset,  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_clip(clip_model, classifier, train_loader, val_loader)
        acc, cm, per_class, inf_time = evaluate_clip(clip_model, classifier, test_loader, "best_clip.pt")
        results["clip"].append(acc)
    
    print(f"\n{'='*50}")
    print("SEED ANALYSIS RESULTS")
    print(f"{'='*50}")
    for model_name, accs, in results.items():
        mean = np.mean(accs)
        std = np.std(accs, ddof = 1)
        print(f"{model_name:15s} | Accuracies: {[f'{a:.4f}' for a in accs]} | Mean: {mean:.4f} | Std: {std:.4f}")

    results_df = pd.DataFrame(results, index=[f"seed_{s}" for s in SEEDS])
    results_df.to_csv("seed_analysis_results.csv")
    print("\nSaved seed_analysis_results.csv")

if __name__ == "__main__":
    main()


    



    
