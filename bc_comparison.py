"""
Controlled comparison of the two Sommerfeld sign conventions.

Runs four fine-tuning jobs from the same saved base weights, at the same
frequency, with the same seed and identical settings. The only thing that varies
is the sign of the boundary residual and whether a dielectric inclusion is present.

    free space  x  current BC   (dEz/dn = +ik Ez)
    free space  x  corrected BC (dEz/dn + ik Ez = 0)
    dielectric  x  current BC
    dielectric  x  corrected BC

Reports, for each pair:
    relative RMS difference between the complex fields
    maximum pointwise difference
    PDE residual and boundary residual, separately
    each field's boundary residual under BOTH conventions
    the free-space vs dielectric separation under each convention

That last quantity is the one that bears on the classification result: if the
separation between the two configurations is unchanged, the sign issue does not
affect the reported high-frequency behaviour.

Run on Newton:
    sbatch run_bc_comparison.sh

Writes bc_comparison_fields.npz, bc_comparison_metrics.csv, and prints a summary.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path

from helmholtz_pinn import (
    build_model, forward, make_plot_grid, laplacian, epsilon_field, Jz,
    sample_points, device,
    HIDDEN, NUM_HIDDEN, OMEGA0, LR, ETA_MIN_FACTOR, GRAD_CLIP,
    LAMBDA_BC, N_INTERIOR, N_BOUNDARY,
)

OUTDIR = Path("pinn_em_results")
OMEGA = 19.515          # dataset index 145, inside the band where errors are systematic
EPOCHS = 3939           # epochs_for_omega(19.515), matching the dataset
SEED = 0
N_EVAL = 201
CIRCLE = (0.0, 0.0, 0.30, 2.0)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_weights(path):
    ckpt = torch.load(path, map_location=device)
    w = [t.to(device).requires_grad_(True) for t in ckpt["weights"]]
    b = [t.to(device).requires_grad_(True) for t in ckpt["biases"]]
    return w, b, ckpt["activations"], ckpt["omegas"]


def sommerfeld_residual(Er_b, Ei_b, coords_b, normals_b, omega, sign):
    """Boundary residual for either convention.

    sign = +1 : dEz/dn = +ik Ez   (as currently implemented)
    sign = -1 : dEz/dn = -ik Ez   (dEz/dn + ik Ez = 0, the outgoing form
                                   for the e^{+i omega t} convention)
    """
    g_r = torch.autograd.grad(Er_b, coords_b, grad_outputs=torch.ones_like(Er_b),
                              create_graph=True)[0]
    g_i = torch.autograd.grad(Ei_b, coords_b, grad_outputs=torch.ones_like(Ei_b),
                              create_graph=True)[0]
    dnEr = (g_r * normals_b).sum(dim=1)
    dnEi = (g_i * normals_b).sum(dim=1)
    k = omega
    bc_r = dnEr + sign * k * Ei_b
    bc_i = dnEi - sign * k * Er_b
    return bc_r.pow(2) + bc_i.pow(2)


def train(weights, biases, activations, omegas, omega, epochs, circle, bc_sign):
    """Fine-tune with the specified boundary convention. Adam only, no L-BFGS,
    matching how the dataset samples were generated."""
    for W in weights:
        W.requires_grad_(True)
    for b in biases:
        b.requires_grad_(True)

    opt = torch.optim.Adam([*weights, *biases], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * ETA_MIN_FACTOR)

    for ep in range(epochs):
        opt.zero_grad()
        xy_i, xy_b, n_b = sample_points(n_interior=N_INTERIOR, n_boundary=N_BOUNDARY)

        out_i = forward(xy_i, weights, biases, activations, omegas)
        Er_i, Ei_i = out_i[:, 0], out_i[:, 1]
        lap_r = laplacian(Er_i, xy_i)
        lap_i = laplacian(Ei_i, xy_i)
        eps_i = epsilon_field(xy_i, circle=circle)
        J = Jz(xy_i)

        res_r = -lap_r - (eps_i * omega**2) * Er_i
        res_i = -lap_i - (eps_i * omega**2) * Ei_i + (omega * J)
        pde_loss = (res_r.pow(2) + res_i.pow(2)).mean()

        out_b = forward(xy_b, weights, biases, activations, omegas)
        bc_loss = sommerfeld_residual(out_b[:, 0], out_b[:, 1], xy_b, n_b,
                                      omega, bc_sign).mean()

        loss = pde_loss + LAMBDA_BC * bc_loss
        loss.backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_([*weights, *biases], max_norm=GRAD_CLIP)
        opt.step()
        sched.step()

        if ep % 500 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:5d} | PDE {pde_loss.item():.3e} | "
                  f"BC {bc_loss.item():.3e}")

    return weights, biases, activations, omegas


def evaluate_field(weights, biases, activations, omegas):
    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, weights, biases, activations, omegas)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()
    return Er + 1j * Ei


def residuals(weights, biases, activations, omegas, omega, circle):
    """PDE residual on a uniform grid, and boundary residual under BOTH
    conventions, reported separately."""
    coords = make_plot_grid(N_EVAL).requires_grad_(True)
    out = forward(coords, weights, biases, activations, omegas)
    Er, Ei = out[:, 0], out[:, 1]
    lap_r = laplacian(Er, coords)
    lap_i = laplacian(Ei, coords)
    eps = epsilon_field(coords, circle=circle)
    J = Jz(coords)
    res_r = -lap_r - (eps * omega**2) * Er
    res_i = -lap_i - (eps * omega**2) * Ei + (omega * J)
    pde_rms = torch.sqrt(torch.mean(res_r**2 + res_i**2)).item()

    _, xy_b, n_b = sample_points(n_interior=100, n_boundary=8000)
    out_b = forward(xy_b, weights, biases, activations, omegas)
    bc_plus = torch.sqrt(sommerfeld_residual(
        out_b[:, 0], out_b[:, 1], xy_b, n_b, omega, +1).mean()).item()
    bc_minus = torch.sqrt(sommerfeld_residual(
        out_b[:, 0], out_b[:, 1], xy_b, n_b, omega, -1).mean()).item()
    return pde_rms, bc_plus, bc_minus


def run_case(label, weight_file, circle, bc_sign):
    print(f"\n=== {label} ===")
    set_seed(SEED)
    w, b, a, o = load_weights(OUTDIR / weight_file)
    w, b, a, o = train(w, b, a, o, OMEGA, EPOCHS, circle, bc_sign)
    field = evaluate_field(w, b, a, o)
    pde, bc_p, bc_m = residuals(w, b, a, o, OMEGA, circle)
    print(f"    PDE residual (rms)          : {pde:.4e}")
    print(f"    BC residual under dn=+ikE   : {bc_p:.4e}")
    print(f"    BC residual under dn=-ikE   : {bc_m:.4e}")
    return field, dict(case=label, pde_residual=pde,
                       bc_residual_plus=bc_p, bc_residual_minus=bc_m)


def compare(a, b, name_a, name_b):
    diff = a - b
    rms_diff = np.sqrt(np.mean(np.abs(diff)**2))
    rms_ref = np.sqrt(np.mean(np.abs(b)**2))
    return dict(
        comparison=f"{name_a} vs {name_b}",
        rms_difference=rms_diff,
        rms_reference=rms_ref,
        relative_rms=rms_diff / rms_ref if rms_ref > 0 else np.nan,
        max_pointwise=np.abs(diff).max(),
    )


def main():
    print(f"omega = {OMEGA}, epochs = {EPOCHS}, seed = {SEED}")
    print("bc_sign +1 = current code, -1 = corrected outgoing condition")

    fields, rows = {}, []

    for tag, wf, circ in [
        ("free_space", "point_source_free_space_weights.pt", None),
        ("dielectric", "point_source_dielectric_sphere_weights.pt", CIRCLE),
    ]:
        for sgn, sname in [(+1, "current"), (-1, "corrected")]:
            key = f"{tag}_{sname}"
            fields[key], row = run_case(key, wf, circ, sgn)
            rows.append(row)

    print("\n" + "=" * 62)
    print("FIELD DIFFERENCES")
    print("=" * 62)

    comps = []
    # effect of the sign change, within each configuration
    for tag in ("free_space", "dielectric"):
        c = compare(fields[f"{tag}_current"], fields[f"{tag}_corrected"],
                    f"{tag} current", f"{tag} corrected")
        comps.append(c)
        print(f"\n{c['comparison']}")
        print(f"  relative rms difference : {c['relative_rms']:.4%}")
        print(f"  max pointwise difference: {c['max_pointwise']:.4e}")

    # the quantity that matters for classification:
    # how far apart are free space and dielectric, under each convention
    print("\n" + "-" * 62)
    print("FREE SPACE vs DIELECTRIC SEPARATION")
    print("-" * 62)
    for sname in ("current", "corrected"):
        c = compare(fields[f"free_space_{sname}"], fields[f"dielectric_{sname}"],
                    f"free_space {sname}", f"dielectric {sname}")
        comps.append(c)
        print(f"\n  under {sname} BC")
        print(f"    rms separation      : {c['rms_difference']:.4e}")
        print(f"    relative separation : {c['relative_rms']:.4%}")

    sep_cur = [c for c in comps if c["comparison"].startswith("free_space current")][0]
    sep_cor = [c for c in comps if c["comparison"].startswith("free_space corrected")][0]
    change = abs(sep_cor["rms_difference"] - sep_cur["rms_difference"])
    rel = change / sep_cur["rms_difference"] if sep_cur["rms_difference"] > 0 else np.nan
    print(f"\n  change in separation from the sign correction: {rel:.4%}")
    print("  (a small value indicates the sign issue does not materially affect")
    print("   the free space vs dielectric distinction the classifier relies on)")

    np.savez("bc_comparison_fields.npz", **fields)
    pd.DataFrame(rows).to_csv("bc_comparison_residuals.csv", index=False)
    pd.DataFrame(comps).to_csv("bc_comparison_differences.csv", index=False)
    print("\nsaved bc_comparison_fields.npz, bc_comparison_residuals.csv, "
          "bc_comparison_differences.csv")


if __name__ == "__main__":
    main()