import torch
import numpy as np
from pathlib import Path

from helmholtz_pinn import (
    train as train_gaussian, forward, make_plot_grid, device, PLOT_N, EARLY_STOP_THR, LAMBDA_BC, N_INTERIOR, N_BOUNDARY
)

from helmholtz_pinn_planewave import train as train_pw

DATASET_DIR = Path("dataset_finetune")
N_EVAL = 201
#FINETUNE_EPOCHS = 500
OUTDIR = Path("pinn_em_results")
#omegas_class1 = [round(4.0 + i * 0.107, 3) for i in range(150)]
TEST_OMEGA = 10.0
epoch_counts = [100, 200, 300, 500, 750, 1000, 1500, 2000]

def load_weights( weight_path ):
    checkpoint = torch.load(weight_path, map_location=device)
    weights = [w.to(device).requires_grad_(True) for w in checkpoint["weights"]]
    biases = [b.to(device).requires_grad_(True) for b in checkpoint["biases"]]
    activations = checkpoint["activations"]
    omegas_net = checkpoint["omegas"]
    return weights, biases, activations, omegas_net

def main():
    weight_path = OUTDIR / "point_source_free_space_weights.pt"

    for epochs in epoch_counts:
        out_file = Path("dataset_finetune") / "epoch_test" / f"test_epoch_{epochs}.npy"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        weights, biases, activations, omegas_net = load_weights(weight_path)

        weights,biases, activations,omegas_net = train_gaussian(
        weights, biases, activations, omegas_net, omega = TEST_OMEGA, 
        epochs = epochs, loss_threshold=EARLY_STOP_THR, lambda_bc=LAMBDA_BC, 
        n_boundary=N_BOUNDARY, use_lbfgs=False, n_interior=N_INTERIOR, 
    )
    
        with torch.no_grad():
            coords = make_plot_grid(N_EVAL)
            out = forward(coords, weights, biases, activations, omegas_net)
            Er = out[:,0].reshape(N_EVAL,N_EVAL).cpu().numpy()
            Ei = out[:,1].reshape(N_EVAL,N_EVAL).cpu().numpy()

        sample = np.stack([Er, Ei], axis=0)
        np.save(out_file, sample)
        print(f"Saved test_epoch_{epochs}.npy")


if __name__ == "__main__":
    main()
