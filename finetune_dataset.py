import torch
import numpy as np
from pathlib import Path
import sys

from helmholtz_pinn import (
    train as train_gaussian, forward, make_plot_grid, device, PLOT_N, EARLY_STOP_THR, LAMBDA_BC, N_INTERIOR, N_BOUNDARY
)

from helmholtz_pinn_planewave import train as train_pw


DATASET_DIR = Path("dataset_finetune")
N_EVAL = 201
FINETUNE_EPOCHS = 2000
OUTDIR = Path("pinn_em_results")
omegas_class1 = [round(4.0 + i * 0.107, 3) for i in range(150)]

def load_weights( weight_path ):
    checkpoint = torch.load(weight_path, map_location=device)
    weights = [w.to(device).requires_grad_(True) for w in checkpoint["weights"]]
    biases = [b.to(device).requires_grad_(True) for b in checkpoint["biases"]]
    activations = checkpoint["activations"]
    omegas_net = checkpoint["omegas"]
    return weights, biases, activations, omegas_net


def epochs_for_omega(omega, base_epochs=2000,max_epochs=4000):
    frac = (omega - 4.0) / (20.0 - 4.0)
    return int(base_epochs + frac * (max_epochs - base_epochs))

def finetune_and_save(scenario_name, omega, sample_id, outdir, weight_path, circle = None, train_fn = train_gaussian):
    outdir.mkdir(parents=True, exist_ok= True)
    out_file = outdir / f"sample_{sample_id:04d}.npy"

    if out_file.exists():
        print(f"Skipping sample_{sample_id:04d}.npy — already exists")
        return
    
    epochs = epochs_for_omega(omega)

    weights,biases,activations,omegas_net = load_weights(weight_path)
    weights,biases, activations,omegas_net = train_fn(
        weights, biases, activations, omegas_net, omega = omega, 
        epochs = epochs, loss_threshold=EARLY_STOP_THR, lambda_bc=LAMBDA_BC, 
        n_boundary=N_BOUNDARY, use_lbfgs=False, n_interior=N_INTERIOR, circle = circle
    )

    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, weights, biases, activations, omegas_net)
        Er = out[:,0].reshape(N_EVAL,N_EVAL).cpu().numpy()
        Ei = out[:,1].reshape(N_EVAL,N_EVAL).cpu().numpy()
    
    sample = np.stack([Er, Ei], axis=0)
    np.save(out_file, sample)
    print(f"[{scenario_name}] saved sample_{sample_id:04d}.npy (omega={omega:.3f})")


def main():
    class_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1


    # Class 1 - free space
    if class_id == 1:
        weight_path = OUTDIR / "point_source_free_space_weights.pt"
        outdir = DATASET_DIR / "free_space_source"
        for i, omega in enumerate(omegas_class1):
            finetune_and_save("free_space_source", omega, i, outdir, weight_path, train_fn=train_gaussian)
        
    # Class 2 - diaelectric
    elif class_id == 2:
        weight_path = OUTDIR / "point_source_dielectric_sphere_weights.pt"
        outdir = DATASET_DIR / "dielectric_source"
        for i, omega in enumerate(omegas_class1):
            finetune_and_save("dielectric_source", omega, i, outdir, weight_path, 
                            circle=(0.0, 0.0, 0.30, 2.0), train_fn=train_gaussian)
    
    # Class 3 - plane wave free space
    elif class_id == 3:
        weight_path = OUTDIR / "incoming_wave_weights.pt"
        outdir = DATASET_DIR / "plane_wave_free"
        for i, omega in enumerate(omegas_class1):
            finetune_and_save("plane_wave_free", omega, i, outdir, weight_path, train_fn=train_pw)
    
    # Class 4 - plane wave diaelectric
    elif class_id == 4:
        weight_path = OUTDIR / "incoming_wave_dielectric_sphere_weights.pt"
        outdir = DATASET_DIR / "plane_wave_dielectric"
        for i, omega in enumerate(omegas_class1):
            finetune_and_save("plane_wave_dielectric", omega, i, outdir, weight_path,
                            circle=(0.0, 0.0, 0.30, 2.0), train_fn=train_pw)

if __name__ == "__main__":
    main()




