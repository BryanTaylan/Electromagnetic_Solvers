import shutil
import csv
from pathlib import Path

class_map = {
    "free_space_source":      ("point_source_free_space", 0),
    "dielectric_source":      ("point_source_dielectric", 1),
    "plane_wave_free":        ("incoming_plane_wave", 2),
    "plane_wave_dielectric":  ("incoming_plane_wave_dielectric", 3),
}

SRC_DIR = Path("dataset_finetune")
DST_DIR = Path("dataset")
omegas = [round(4.0 + i * 0.107, 3) for i in range(150)]

def main():

    for old_name, (new_name, label) in class_map.items():
        (DST_DIR / new_name).mkdir(parents=True, exist_ok=True)

    metadata = [["filepath", "class_label", "frequency"]]

    for old_name, (new_name, label) in class_map.items():
        src_folder = SRC_DIR / old_name
        dst_folder = DST_DIR / new_name

        for i, omega in enumerate(omegas):

            src_file = src_folder / f"sample_{i:04d}.npy"

            dst_file = dst_folder / f"sample_w{omega:.3f}_{i:04d}.npy"

            shutil.copy(src_file, dst_file)

            metadata.append([str(dst_file), label, omega])

            print(f"Copied {src_file.name} -> {dst_file.name}")

    with open(DST_DIR / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metadata)
    
    print("Done! metadata.csv saved.")

if __name__ == "__main__":
    main()

