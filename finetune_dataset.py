import torch
import numpy as np
from pathlib import Path

from helmholtz_pinn import (
    train, forward, make_plot_grid, device, PLOT_N, EARLY_STOP_THR, LAMBDA_BC, N_INTERIOR, N_BOUNDARY
)

DATASET_DIR = Path("dataset_finetune")
N_EVAL = 201
FINETUNE_EPOCHS = 500
OUTDIR = Path("pinn_em_results")
omegas_class1 = [round(4.0 + i * 0.107, 3) for i in range(150)]

def load_weights( weight_path ):
    checkpoint = torch.load(weight_path, map_location=device)
    weights = [w.to(device).requires_grad_(True) for w in checkpoint["weights"]]
    biases = [b.to(device).requires_grad_(True) for b in checkpoint["biases"]]
    activations = checkpoint["activations"]
    omegas_net = checkpoint["omegas"]
    return weights, biases, activations, omegas_net

def finetune_and_save(scenario_name, omega, sample_id, outdir, weight_path, circle = None, epochs = FINETUNE_EPOCHS):
    outdir.mkdir(parents=True, exist_ok= True)
    out_file = outdir / f"sample_{sample_id:04d}.npy"

    if out_file.exists():
        print(f"Skipping sample_{sample_id:04d}.npy — already exists")
        return

    weights,biases,activations,omegas_net = load_weights(weight_path)
    weights,biases, activations,omegas_net = train(
        weights, biases, activations, omegas_net, omega = omega, 
        epochs = epochs, loss_threshold=EARLY_STOP_THR, lambda_bc=LAMBDA_BC, 
        n_boundary=N_BOUNDARY, use_lbfgs=False, n_interior=N_INTERIOR
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
    weight_path = OUTDIR / "point_source_free_space_weights.pt"
    if not weight_path.exists():
        print(f"ERROR: {weight_path} not found. Run helmholtz_pinn.py first.")
        return

    outdir = DATASET_DIR / "free_space_source"
    for i, omega in enumerate(omegas_class1):
        finetune_and_save("free_space_source", omega, i, outdir, weight_path)

if __name__ == "__main__":
    main()




