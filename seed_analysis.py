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
SEEDS = [0, 1, 2, 3, 4]
NUM_CLASSES = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_sample_predictions(model, test_loader, test_df, checkpoint,
                           model_name, seed, clip_model=None):
    """Return one record per test sample: true label, predicted label, frequency.

    IMPORTANT: test_loader must be built with shuffle=False so that batch order
    matches the row order of test_df. Do not change this.
    """
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            if clip_model is not None:
                feats = clip_model.encode_image(x).float()
                logits = model(feats)
            else:
                logits = model(x)
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    assert len(preds) == len(test_df), (
        f"prediction count {len(preds)} != test_df rows {len(test_df)}; "
        "check that test_loader uses shuffle=False"
    )

    records = []
    for i, p in enumerate(preds):
        row = test_df.iloc[i]
        records.append({
            "model": model_name,
            "seed": seed,
            "filepath": row["filepath"],
            "frequency": float(row["frequency"]),
            "true_label": int(row["class_label"]),
            "pred_label": int(p),
            "correct": int(p) == int(row["class_label"]),
        })
    return records


def benchmark_inference(model, test_loader, clip_model=None, n_warmup=2):
    """Measure mean inference time per sample under controlled conditions."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            for x, _ in test_loader:
                x = x.to(device)
                if clip_model is not None:
                    model(clip_model.encode_image(x).float())
                else:
                    model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        n = 0
        for x, _ in test_loader:
            x = x.to(device)
            if clip_model is not None:
                model(clip_model.encode_image(x).float())
            else:
                model(x)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    return elapsed / n


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    train_df = pd.read_csv(DATASET_DIR / "train_split.csv")
    val_df = pd.read_csv(DATASET_DIR / "val_split.csv")
    test_df = pd.read_csv(DATASET_DIR / "test_split.csv")

    train_df_224 = pd.read_csv(DATASET_224_DIR / "train_split.csv")
    val_df_224 = pd.read_csv(DATASET_224_DIR / "val_split.csv")
    test_df_224 = pd.read_csv(DATASET_224_DIR / "test_split.csv")

    model_names = ["cnn", "resnet_probe", "resnet_ft", "resnet_scratch", "clip"]

    results = {m: [] for m in model_names}
    confusions = {m: [] for m in model_names}
    inference_times = {m: [] for m in model_names}
    param_counts = {}
    all_records = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        # ---------------- Baseline CNN ----------------
        print("\n--- Baseline CNN ---")
        train_loader = DataLoader(FieldDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(FieldDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(FieldDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

        ckpt = f"best_cnn_seed{seed}.pt"
        model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        param_counts["cnn"] = count_trainable(model)
        train_model(model, train_loader, val_loader)
        # train_model writes to best_cnn.pt; preserve a per-seed copy
        torch.save(torch.load("best_cnn.pt", map_location=device), ckpt)

        acc, cm, per_class, _ = evaluate_cnn(model, test_loader, checkpoint=ckpt)
        results["cnn"].append(acc)
        confusions["cnn"].append(cm)
        inference_times["cnn"].append(benchmark_inference(model, test_loader))
        all_records += per_sample_predictions(
            model, test_loader, test_df, ckpt, "cnn", seed)

        # ---------------- shared 224 loaders ----------------
        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        train_loader_224 = DataLoader(FieldDataset224(train_df_224, transform=transform),
                                      batch_size=BATCH_SIZE, shuffle=True)
        val_loader_224 = DataLoader(FieldDataset224(val_df_224, transform=transform),
                                    batch_size=BATCH_SIZE, shuffle=False)
        test_loader_224 = DataLoader(FieldDataset224(test_df_224, transform=transform),
                                     batch_size=BATCH_SIZE, shuffle=False)

        # ---------------- ResNet-18 linear probe ----------------
        print("\n--- ResNet-18 Linear Probe ---")
        ckpt = f"best_resnet_probe_seed{seed}.pt"
        model = resnet18(weights=weights).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(512, NUM_CLASSES).to(device)
        param_counts["resnet_probe"] = count_trainable(model)
        train_linear_probe(model, train_loader_224, val_loader_224)
        torch.save(torch.load("best_resnet_probe.pt", map_location=device), ckpt)

        acc, cm, per_class, _ = evaluate_resnet(model, test_loader_224, ckpt)
        results["resnet_probe"].append(acc)
        confusions["resnet_probe"].append(cm)
        inference_times["resnet_probe"].append(benchmark_inference(model, test_loader_224))
        all_records += per_sample_predictions(
            model, test_loader_224, test_df_224, ckpt, "resnet_probe", seed)

        # ---------------- ResNet-18 full fine-tune ----------------
        print("\n--- ResNet-18 Fine-tune ---")
        ckpt = f"best_resnet_finetune_seed{seed}.pt"
        model = resnet18(weights=weights).to(device)
        model.fc = nn.Linear(512, NUM_CLASSES).to(device)
        for p in model.parameters():
            p.requires_grad = True
        param_counts["resnet_ft"] = count_trainable(model)
        train_finetune(model, train_loader_224, val_loader_224,
                       checkpoint="best_resnet_finetune.pt")
        torch.save(torch.load("best_resnet_finetune.pt", map_location=device), ckpt)

        acc, cm, per_class, _ = evaluate_resnet(model, test_loader_224, ckpt)
        results["resnet_ft"].append(acc)
        confusions["resnet_ft"].append(cm)
        inference_times["resnet_ft"].append(benchmark_inference(model, test_loader_224))
        all_records += per_sample_predictions(
            model, test_loader_224, test_df_224, ckpt, "resnet_ft", seed)

        # ---------------- ResNet-18 from scratch ----------------
        print("\n--- ResNet-18 From Scratch ---")
        ckpt = f"best_resnet_scratch_seed{seed}.pt"
        model = resnet18(weights=None).to(device)
        model.fc = nn.Linear(512, NUM_CLASSES).to(device)
        param_counts["resnet_scratch"] = count_trainable(model)
        train_finetune(model, train_loader_224, val_loader_224,
                       checkpoint="best_resnet_scratch.pt")
        torch.save(torch.load("best_resnet_scratch.pt", map_location=device), ckpt)

        acc, cm, per_class, _ = evaluate_resnet(model, test_loader_224, ckpt)
        results["resnet_scratch"].append(acc)
        confusions["resnet_scratch"].append(cm)
        inference_times["resnet_scratch"].append(benchmark_inference(model, test_loader_224))
        all_records += per_sample_predictions(
            model, test_loader_224, test_df_224, ckpt, "resnet_scratch", seed)

        # ---------------- OpenCLIP ----------------
        print("\n--- OpenCLIP ---")
        ckpt = f"best_clip_seed{seed}.pt"
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        classifier = CLIPClassifier(feature_dim=512, num_classes=NUM_CLASSES).to(device)
        param_counts["clip"] = count_trainable(classifier)

        train_loader_clip = DataLoader(FieldDataset224(train_df_224, transform=preprocess),
                                       batch_size=BATCH_SIZE, shuffle=True)
        val_loader_clip = DataLoader(FieldDataset224(val_df_224, transform=preprocess),
                                     batch_size=BATCH_SIZE, shuffle=False)
        test_loader_clip = DataLoader(FieldDataset224(test_df_224, transform=preprocess),
                                      batch_size=BATCH_SIZE, shuffle=False)

        train_clip(clip_model, classifier, train_loader_clip, val_loader_clip)
        torch.save(torch.load("best_clip.pt", map_location=device), ckpt)

        acc, cm, per_class, _ = evaluate_clip(clip_model, classifier, test_loader_clip, ckpt)
        results["clip"].append(acc)
        confusions["clip"].append(cm)
        inference_times["clip"].append(
            benchmark_inference(classifier, test_loader_clip, clip_model=clip_model))
        all_records += per_sample_predictions(
            classifier, test_loader_clip, test_df_224, ckpt, "clip", seed,
            clip_model=clip_model)

    # ================= SUMMARY =================
    print(f"\n{'='*60}")
    print("SEED ANALYSIS RESULTS")
    print(f"{'='*60}")

    summary_rows = []
    for m in model_names:
        accs = np.array(results[m])
        cm_mean = np.mean(np.stack(confusions[m]), axis=0)
        per_class_mean = cm_mean.diagonal() / cm_mean.sum(axis=1)
        inf_ms = np.mean(inference_times[m]) * 1000

        print(f"\n{m}")
        print(f"  accuracies : {[f'{a:.4f}' for a in accs]}")
        print(f"  mean +/- std: {accs.mean():.4f} +/- {accs.std(ddof=1):.4f}")
        print(f"  per-class (mean over seeds): "
              f"{[f'{p*100:.1f}%' for p in per_class_mean]}")
        print(f"  trainable params: {param_counts[m]:,}")
        print(f"  inference: {inf_ms:.2f} ms/sample")
        print(f"  mean confusion matrix:\n{np.round(cm_mean, 2)}")

        summary_rows.append({
            "model": m,
            "mean_acc": accs.mean(),
            "std_acc": accs.std(ddof=1),
            **{f"per_class_{i}": per_class_mean[i] for i in range(NUM_CLASSES)},
            "trainable_params": param_counts[m],
            "inference_ms": inf_ms,
        })

    pd.DataFrame(results, index=[f"seed_{s}" for s in SEEDS]).to_csv(
        "seed_analysis_results.csv")
    pd.DataFrame(summary_rows).to_csv("seed_analysis_summary.csv", index=False)

    records_df = pd.DataFrame(all_records)
    records_df.to_csv("per_sample_predictions.csv", index=False)

    np.savez("confusion_matrices.npz",
             **{m: np.stack(confusions[m]) for m in model_names})

    # ---- which samples are misclassified, and how often ----
    errors = records_df[~records_df["correct"]]
    if len(errors):
        print(f"\n{'='*60}")
        print("MISCLASSIFIED SAMPLES BY MODEL AND FREQUENCY")
        print(f"{'='*60}")
        for m in model_names:
            sub = errors[errors["model"] == m]
            if not len(sub):
                print(f"\n{m}: no errors")
                continue
            print(f"\n{m}: {len(sub)} errors across {len(SEEDS)} seeds")
            grp = (sub.groupby(["frequency", "true_label", "pred_label"])
                      .size().reset_index(name="n_seeds")
                      .sort_values("frequency"))
            for _, r in grp.iterrows():
                print(f"  omega={r['frequency']:.3f}  true={r['true_label']} "
                      f"pred={r['pred_label']}  missed in {r['n_seeds']}/{len(SEEDS)} seeds")

    print("\nSaved: seed_analysis_results.csv, seed_analysis_summary.csv, "
          "per_sample_predictions.csv, confusion_matrices.npz")


if __name__ == "__main__":
    main()