import torch
import numpy as np
from pathlib import Path

from pinn_core import (
    build_model, forward, make_plot_grid,
    HIDDEN, NUM_HIDDEN, OMEGA0, PLOT_N, device,
    EARLY_STOP_THR, N_INTERIOR, N_BOUNDARY
)
from helmholtz_pinn_planewave import train as train_pw

DATASET_DIR = Path("dataset")
N_EVAL = 201
GEN_EPOCHS = 5000
LAMBDA_DIR = 50.0

# Class 4 — plane_wave_dielectric: plane wave + circle
# Varying omega, circle radius, circle epsilon
plane_wave_dielectric_params = [
    {"omega": 8,  "circle": (0.0, 0.0, 0.2, 1.5)},
    {"omega": 8,  "circle": (0.0, 0.0, 0.3, 2.0)},
    {"omega": 8,  "circle": (0.0, 0.0, 0.4, 3.0)},
    {"omega": 10, "circle": (0.0, 0.0, 0.2, 2.0)},
    {"omega": 10, "circle": (0.0, 0.0, 0.3, 3.0)},
    {"omega": 10, "circle": (0.0, 0.0, 0.4, 1.5)},
    {"omega": 12, "circle": (0.0, 0.0, 0.2, 3.0)},
    {"omega": 12, "circle": (0.0, 0.0, 0.3, 1.5)},
    {"omega": 14, "circle": (0.0, 0.0, 0.4, 2.0)},
    {"omega": 14, "circle": (0.0, 0.0, 0.2, 1.5)},
    {"omega": 16, "circle": (0.0, 0.0, 0.3, 3.0)},
    {"omega": 16, "circle": (0.0, 0.0, 0.4, 2.0)},
]


def generate_sample(scenario_name, params, sample_id, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    omega  = params["omega"]
    circle = params["circle"]

    weights, biases, activations, omegas_net = build_model(
        input_dim=2, hidden_dim=HIDDEN, output_dim=2,
        num_hidden=NUM_HIDDEN, omega_0=OMEGA0
    )
    weights, biases, activations, omegas_net = train_pw(
        weights, biases, activations, omegas_net,
        omega=omega, epochs=GEN_EPOCHS,
        loss_threshold=EARLY_STOP_THR,
        lambda_bc=LAMBDA_DIR,
        circle=circle,
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
    print(f"[{scenario_name}] saved sample_{sample_id:04d}.npy "
          f"(omega={omega}, circle={circle})")


def main():
    outdir = DATASET_DIR / "plane_wave_dielectric"
    for i, params in enumerate(plane_wave_dielectric_params):
        generate_sample("plane_wave_dielectric", params, i, outdir)


if __name__ == "__main__":
    main()
