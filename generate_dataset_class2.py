import torch
import numpy as np
from pathlib import Path

from pinn_core import (
    build_model, train, forward, make_plot_grid,
    HIDDEN, NUM_HIDDEN, OMEGA0, PLOT_N, device,
    EARLY_STOP_THR, LAMBDA_BC, N_INTERIOR, N_BOUNDARY
)

DATASET_DIR = Path("dataset")
N_EVAL = 201
GEN_EPOCHS = 5000

# Class 2 — dielectric_source: Gaussian source + circle
# Varying omega, source center, circle radius, circle epsilon
dielectric_source_params = [
    {"omega": 8,  "center": (0.0,  0.0),  "circle": (0.0, 0.0, 0.2, 1.5)},
    {"omega": 8,  "center": (0.3,  0.0),  "circle": (0.0, 0.0, 0.3, 2.0)},
    {"omega": 8,  "center": (-0.3, 0.0),  "circle": (0.0, 0.0, 0.4, 3.0)},
    {"omega": 10, "center": (0.0,  0.0),  "circle": (0.0, 0.0, 0.2, 2.0)},
    {"omega": 10, "center": (0.5,  0.0),  "circle": (0.0, 0.0, 0.3, 1.5)},
    {"omega": 10, "center": (0.0,  0.3),  "circle": (0.0, 0.0, 0.4, 2.0)},
    {"omega": 12, "center": (0.0,  0.0),  "circle": (0.0, 0.0, 0.3, 3.0)},
    {"omega": 12, "center": (0.5,  0.3),  "circle": (0.0, 0.0, 0.2, 2.0)},
    {"omega": 14, "center": (0.0,  0.0),  "circle": (0.0, 0.0, 0.4, 1.5)},
    {"omega": 14, "center": (-0.5, 0.0),  "circle": (0.0, 0.0, 0.3, 2.0)},
    {"omega": 16, "center": (0.0,  0.0),  "circle": (0.0, 0.0, 0.2, 3.0)},
    {"omega": 16, "center": (0.3, -0.3),  "circle": (0.0, 0.0, 0.4, 2.0)},
]


def generate_sample(scenario_name, params, sample_id, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    omega  = params["omega"]
    circle = params.get("circle", None)
    center = params.get("center", (0.0, 0.0))

    weights, biases, activations, omegas_net = build_model(
        input_dim=2, hidden_dim=HIDDEN, output_dim=2,
        num_hidden=NUM_HIDDEN, omega_0=OMEGA0
    )
    weights, biases, activations, omegas_net = train(
        weights, biases, activations, omegas_net,
        omega=omega, epochs=GEN_EPOCHS,
        loss_threshold=EARLY_STOP_THR, lambda_bc=LAMBDA_BC,
        circle=circle, n_interior=N_INTERIOR, n_boundary=N_BOUNDARY,
        use_lbfgs=True, source_center=center,
    )

    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, weights, biases, activations, omegas_net)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()

    sample = np.stack([Er, Ei], axis=0)
    np.save(outdir / f"sample_{sample_id:04d}.npy", sample)
    print(f"[{scenario_name}] saved sample_{sample_id:04d}.npy "
          f"(omega={omega}, center={center}, circle={circle})")


def main():
    outdir = DATASET_DIR / "dielectric_source"
    for i, params in enumerate(dielectric_source_params):
        generate_sample("dielectric_source", params, i, outdir)


if __name__ == "__main__":
    main()
