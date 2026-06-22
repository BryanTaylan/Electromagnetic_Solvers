import torch
import numpy as np
from pathlib import Path

from pinn_core import (
    build_model, forward, make_plot_grid,
    HIDDEN, NUM_HIDDEN, OMEGA0, PLOT_N, device,
    EARLY_STOP_THR, N_INTERIOR, N_BOUNDARY
)
from helmholtz_pinn_planewave import train as train_pw, planewave_bc_loss, sommerfeld_bc_loss

DATASET_DIR = Path("dataset")
N_EVAL = 201
GEN_EPOCHS = 5000
LAMBDA_DIR = 50.0

# Class 3 — plane_wave_free: plane wave, no circle
# Varying omega only
plane_wave_free_params = [
    {"omega": 8},
    {"omega": 9},
    {"omega": 10},
    {"omega": 11},
    {"omega": 12},
    {"omega": 13},
    {"omega": 14},
    {"omega": 15},
    {"omega": 16},
    {"omega": 17},
    {"omega": 18},
    {"omega": 19},
]


def generate_sample(scenario_name, params, sample_id, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    omega = params["omega"]

    weights, biases, activations, omegas_net = build_model(
        input_dim=2, hidden_dim=HIDDEN, output_dim=2,
        num_hidden=NUM_HIDDEN, omega_0=OMEGA0
    )
    weights, biases, activations, omegas_net = train_pw(
        weights, biases, activations, omegas_net,
        omega=omega, epochs=GEN_EPOCHS,
        loss_threshold=EARLY_STOP_THR,
        lambda_bc=LAMBDA_DIR,
        circle=None,
        n_interior=N_INTERIOR, n_boundary=N_BOUNDARY,
        use_lbfgs=True,
    )

    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, weights, biases, activations, omegas_net)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()

    sample = np.stack([Er, Ei], axis=0)
    np.save(outdir / f"sample_{sample_id:04d}.npy", sample)
    print(f"[{scenario_name}] saved sample_{sample_id:04d}.npy (omega={omega})")


def main():
    outdir = DATASET_DIR / "plane_wave_free"
    for i, params in enumerate(plane_wave_free_params):
        generate_sample("plane_wave_free", params, i, outdir)


if __name__ == "__main__":
    main()
