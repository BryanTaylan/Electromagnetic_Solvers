"""
Convergence study at omega = 19.515 for the two point-source configurations.

Uses the CORRECTED outgoing boundary condition throughout,
    dEz/dn + ik Ez = 0,
and holds everything else fixed. Each configuration is trained in a single run,
with the field and residuals recorded at 2000, 4000, 8000 and 16000 Adam epochs,
and again after an L-BFGS refinement.

At every stage this records:
    PDE residual and boundary residual, separately
    the free space to dielectric separation, using the same metric as before
    how much each field changed since the previous stage

The question is whether the fields and their separation stabilise as the residual
falls. Stabilisation would support the interpretation that the original
high-frequency samples were under converged. A residual that falls substantially
while the separation does not move would point elsewhere.

Run on Newton:
    sbatch run_convergence_study.sh

Writes convergence_fields.npz and convergence_metrics.csv.
"""

import numpy as np
import torch
import pandas as pd

from helmholtz_pinn import (
    forward, make_plot_grid, laplacian, epsilon_field, Jz, sample_points, device,
    LR, ETA_MIN_FACTOR, GRAD_CLIP, LAMBDA_BC, N_INTERIOR, N_BOUNDARY,
)
from pathlib import Path

OUTDIR = Path("pinn_em_results")
OMEGA = 19.515
STAGES = [2000, 4000, 8000, 16000]      # cumulative Adam epochs
SEED = 0
N_EVAL = 201
CIRCLE = (0.0, 0.0, 0.30, 2.0)
BC_SIGN = -1                            # corrected: dEz/dn + ik Ez = 0


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
    g_r = torch.autograd.grad(Er_b, coords_b, grad_outputs=torch.ones_like(Er_b),
                              create_graph=True)[0]
    g_i = torch.autograd.grad(Ei_b, coords_b, grad_outputs=torch.ones_like(Ei_b),
                              create_graph=True)[0]
    dnEr = (g_r * normals_b).sum(dim=1)
    dnEi = (g_i * normals_b).sum(dim=1)
    k = omega
    return (dnEr + sign * k * Ei_b).pow(2) + (dnEi - sign * k * Er_b).pow(2)


def losses(weights, biases, activations, omegas, omega, circle, xy_i, xy_b, n_b):
    out_i = forward(xy_i, weights, biases, activations, omegas)
    Er_i, Ei_i = out_i[:, 0], out_i[:, 1]
    lap_r = laplacian(Er_i, xy_i)
    lap_i = laplacian(Ei_i, xy_i)
    eps_i = epsilon_field(xy_i, circle=circle)
    J = Jz(xy_i)
    res_r = -lap_r - (eps_i * omega**2) * Er_i
    res_i = -lap_i - (eps_i * omega**2) * Ei_i + (omega * J)
    pde = (res_r.pow(2) + res_i.pow(2)).mean()

    out_b = forward(xy_b, weights, biases, activations, omegas)
    bc = sommerfeld_residual(out_b[:, 0], out_b[:, 1], xy_b, n_b,
                             omega, BC_SIGN).mean()
    return pde, bc


def evaluate_field(weights, biases, activations, omegas):
    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, weights, biases, activations, omegas)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()
    return Er + 1j * Ei


def measure(weights, biases, activations, omegas, omega, circle):
    """Grid-evaluated PDE residual and boundary residual, separately."""
    coords = make_plot_grid(N_EVAL).requires_grad_(True)
    out = forward(coords, weights, biases, activations, omegas)
    Er, Ei = out[:, 0], out[:, 1]
    lap_r = laplacian(Er, coords)
    lap_i = laplacian(Ei, coords)
    eps = epsilon_field(coords, circle=circle)
    J = Jz(coords)
    res_r = -lap_r - (eps * omega**2) * Er
    res_i = -lap_i - (eps * omega**2) * Ei + (omega * J)
    pde = torch.sqrt(torch.mean(res_r**2 + res_i**2)).item()

    _, xy_b, n_b = sample_points(n_interior=100, n_boundary=8000)
    out_b = forward(xy_b, weights, biases, activations, omegas)
    bc = torch.sqrt(sommerfeld_residual(
        out_b[:, 0], out_b[:, 1], xy_b, n_b, omega, BC_SIGN).mean()).item()
    return pde, bc


def train_with_stages(weight_file, circle, label):
    print(f"\n{'='*62}\n{label}\n{'='*62}")
    set_seed(SEED)
    w, b, a, o = load_weights(OUTDIR / weight_file)

    for t in w:
        t.requires_grad_(True)
    for t in b:
        t.requires_grad_(True)

    total = STAGES[-1]
    opt = torch.optim.Adam([*w, *b], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total, eta_min=LR * ETA_MIN_FACTOR)

    fields, rows = {}, []
    for ep in range(total):
        opt.zero_grad()
        xy_i, xy_b, n_b = sample_points(n_interior=N_INTERIOR,
                                        n_boundary=N_BOUNDARY)
        pde, bc = losses(w, b, a, o, OMEGA, circle, xy_i, xy_b, n_b)
        loss = pde + LAMBDA_BC * bc
        loss.backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_([*w, *b], max_norm=GRAD_CLIP)
        opt.step()
        sched.step()

        if (ep + 1) in STAGES:
            pde_g, bc_g = measure(w, b, a, o, OMEGA, circle)
            fields[f"{label}_{ep+1}"] = evaluate_field(w, b, a, o)
            rows.append(dict(case=label, stage=str(ep + 1),
                             adam_epochs=ep + 1,
                             pde_residual=pde_g, bc_residual=bc_g))
            print(f"  after {ep+1:>6} epochs | PDE {pde_g:.4e} | BC {bc_g:.4e}")

    # L-BFGS refinement
    print("  running L-BFGS refinement...")
    lbfgs = torch.optim.LBFGS([*w, *b], lr=1.0, max_iter=500,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        xy_i, xy_b, n_b = sample_points(n_interior=N_INTERIOR,
                                        n_boundary=N_BOUNDARY)
        pde, bc = losses(w, b, a, o, OMEGA, circle, xy_i, xy_b, n_b)
        L = pde + LAMBDA_BC * bc
        L.backward()
        return L

    lbfgs.step(closure)
    pde_g, bc_g = measure(w, b, a, o, OMEGA, circle)
    fields[f"{label}_lbfgs"] = evaluate_field(w, b, a, o)
    rows.append(dict(case=label, stage="lbfgs", adam_epochs=total,
                     pde_residual=pde_g, bc_residual=bc_g))
    print(f"  after L-BFGS       | PDE {pde_g:.4e} | BC {bc_g:.4e}")

    return fields, rows


def rms(z):
    return float(np.sqrt(np.mean(np.abs(z) ** 2)))


def main():
    print(f"omega = {OMEGA}, corrected BC (dEz/dn + ik Ez = 0), seed = {SEED}")
    print(f"stages: {STAGES} Adam epochs, then L-BFGS")

    f1, r1 = train_with_stages("point_source_free_space_weights.pt",
                               None, "free_space")
    f2, r2 = train_with_stages("point_source_dielectric_sphere_weights.pt",
                               CIRCLE, "dielectric")
    fields = {**f1, **f2}
    rows = r1 + r2

    stage_names = [str(s) for s in STAGES] + ["lbfgs"]

    print("\n" + "=" * 62)
    print("FIELD STABILISATION")
    print("=" * 62)
    print(f"\n{'case':<12}{'stage':>8}{'rms |Ez|':>12}{'change vs prev':>16}")
    for case in ("free_space", "dielectric"):
        prev = None
        for s in stage_names:
            z = fields[f"{case}_{s}"]
            if prev is None:
                chg = "--"
            else:
                chg = f"{rms(z - prev) / rms(z):.2%}"
            print(f"{case:<12}{s:>8}{rms(z):>12.4e}{chg:>16}")
            prev = z
        print()

    print("=" * 62)
    print("FREE SPACE TO DIELECTRIC SEPARATION")
    print("=" * 62)
    print(f"\n{'stage':>8}{'separation':>14}{'relative':>12}{'change vs prev':>16}")
    prev_sep = None
    for s in stage_names:
        a, b = fields[f"free_space_{s}"], fields[f"dielectric_{s}"]
        sep = rms(a - b)
        rel = sep / rms(b)
        chg = "--" if prev_sep is None else f"{abs(sep - prev_sep)/prev_sep:.2%}"
        print(f"{s:>8}{sep:>14.4e}{rel:>11.1%}{chg:>16}")
        rows.append(dict(case="separation", stage=s, adam_epochs=np.nan,
                         separation=sep, separation_relative=rel))
        prev_sep = sep

    print("\nIf the residual falls while the fields and the separation settle,")
    print("that supports the under-convergence reading. If the residual falls")
    print("substantially but the separation does not move, it points elsewhere.")

    np.savez("convergence_fields.npz", **fields)
    pd.DataFrame(rows).to_csv("convergence_metrics.csv", index=False)
    print("\nwrote convergence_fields.npz and convergence_metrics.csv")


if __name__ == "__main__":
    main()