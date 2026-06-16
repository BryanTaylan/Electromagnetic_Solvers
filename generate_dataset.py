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
GEN_EPOCHS = 5000   # reduced epochs for dataset generation speed

# ---------------------------------------------------------
# Parameter sets for Class 1: free_space_source
# Varying omega and source center (cx, cy)
# ---------------------------------------------------------
free_space_params = [
    {"omega": 8,  "center": (0.0, 0.0)},
    {"omega": 8,  "center": (0.3, 0.0)},
    {"omega": 8,  "center": (-0.3, 0.0)},
    {"omega": 10, "center": (0.0, 0.0)},
    {"omega": 10, "center": (0.5, 0.0)},
    {"omega": 10, "center": (0.0, 0.3)},
    {"omega": 12, "center": (0.0, 0.0)},
    {"omega": 12, "center": (0.5, 0.3)},
    {"omega": 14, "center": (0.0, 0.0)},
    {"omega": 14, "center": (-0.5, 0.0)},
    {"omega": 16, "center": (0.0, 0.0)},
    {"omega": 16, "center": (0.3, -0.3)},
]


def generate_sample(scenario_name, params, sample_id, outdir, circle=None):
    """Train one PINN with given params and save the field as a (2, N, N) array."""
    outdir.mkdir(parents=True, exist_ok=True)

    omega = params["omega"]

    weights, biases, activations, omegas_net = build_model(
        input_dim=2, hidden_dim=HIDDEN, output_dim=2,
        num_hidden=NUM_HIDDEN, omega_0=OMEGA0
    )

    weights, biases, activations, omegas_net = train(
        weights, biases, activations, omegas_net,
        omega=omega,
        epochs=GEN_EPOCHS,
        loss_threshold=EARLY_STOP_THR,
        lambda_bc=LAMBDA_BC,
        circle=circle,
        n_interior=N_INTERIOR,
        n_boundary=N_BOUNDARY,
        use_lbfgs=True,
        source_center=params.get("center", (0.0, 0.0)),
    )

    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, weights, biases, activations, omegas_net)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()

    sample = np.stack([Er, Ei], axis=0)  # shape (2, 201, 201)
    np.save(outdir / f"sample_{sample_id:04d}.npy", sample)
    print(f"[{scenario_name}] saved sample_{sample_id:04d}.npy "
          f"(omega={omega}, center={params.get('center')})")


def main():
    outdir = DATASET_DIR / "free_space_source"
    for i, params in enumerate(free_space_params):
        generate_sample("free_space_source", params, i, outdir, circle=None)


if __name__ == "__main__":
    main()