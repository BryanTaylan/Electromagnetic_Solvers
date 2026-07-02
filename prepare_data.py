import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd

SRC_DIR = Path("dataset")
DST_DIR = Path("dataset_224")
TARGET_SIZE = 224


def process_sample(src_path, dst_path):
    sample = np.load(src_path).astype(np.float32)

    Er = sample[0]
    Ei = sample[1]

    E_mag = np.sqrt(Er**2 + Ei**2)

    sample_3ch = np.stack([Er,Ei,E_mag], axis=0)

    for c in range(3):
        max_val = np.abs(sample_3ch[c]).max()
        if max_val > 0:
            sample_3ch[c] = sample_3ch[c] / max_val
    
    tensor = torch.tensor(sample_3ch).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear')
    sample_224 = tensor.squeeze(0).numpy()

    dst_path.parent.mkdir(parents=True, exist_ok = True)
    np.save(dst_path, sample_224)

def main():

    train_df = pd.read_csv(SRC_DIR/ "train_split.csv")
    val_df = pd.read_csv(SRC_DIR/ "val_split.csv")
    test_df = pd.read_csv(SRC_DIR / "test_split.csv")

    all_df = pd.concat([train_df,val_df,test_df]).drop_duplicates(subset="filepath")

    for _, row in all_df.iterrows():
        src_path = Path(row["filepath"])
        dst_path = DST_DIR / src_path.relative_to(SRC_DIR)

        if dst_path.exists():
            print(f"Skipping {dst_path.name} — already exists")
            continue

        process_sample(src_path, dst_path)        
        print(f"Processed {src_path.name}")
    
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df["filepath"] = df["filepath"].str.replace("dataset/", "dataset_224/", regex=False)
        df.to_csv(DST_DIR/ f"{name}_split.csv", index=False)
    
    print("Done! dataset_224 ready.")


if __name__ == "__main__":
    main()


