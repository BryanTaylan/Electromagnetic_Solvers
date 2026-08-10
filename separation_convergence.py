"""
Does the free space to dielectric separation converge with training length?

The earlier convergence study checkpointed a single 16000-epoch run at
intermediate points. Those checkpoints share one cosine schedule, so they were
sampled mid-anneal at high learning rate and are not equivalent to standalone
runs of that length. This script fixes that: each budget gets its own
independent run with its own schedule annealing over its full length, exactly
as the dataset samples were generated.

Budgets: 4000, 8000, 16000, 32000 Adam epochs, each followed by L-BFGS.
Two configurations, so eight runs.

The question is whether the separation approaches a stable value. Two fully
annealed points were already available, 3939 and 16000 epochs, and the
separation grew 51% between them. If it keeps growing at the same rate through
32000, the separation has not converged and the dataset understates it by an
unknown amount. If it flattens, we can state the converged value.

Run on Newton:
    sbatch run_separation_convergence.sh
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path

from helmholtz_pinn import (
    forward, make_plot_grid, laplacian, epsilon_field, Jz, sample_points, device,
    LR, ETA_MIN_FACTOR, GRAD_CLIP, LAMBDA_BC, N_INTERIOR, N_BOUNDARY,
)

OUTDIR = Path("pinn_em_results")
OMEGA = 19.515
BUDGETS = [4000, 8000, 16000, 32000]
SEED = 0
N_EVAL = 201
CIRCLE = (0.0, 0.0, 0.30, 2.0)
BC_SIGN = -1                       # corrected outgoing condition


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


def bc_residual(Er_b, Ei_b, coords_b, normals_b, omega, sign=BC_SIGN):
    g_r = torch.autograd.grad(Er_b, coords_b, grad_outputs=torch.ones_like(Er_b),
                              create_graph=True)[0]
    g_i = torch.autograd.grad(Ei_b, coords_b, grad_outputs=torch.ones_like(Ei_b),
                              create_graph=True)[0]
    dnEr = (g_r * normals_b).sum(dim=1)
    dnEi = (g_i * normals_b).sum(dim=1)
    k = omega
    return (dnEr + sign * k * Ei_b).pow(2) + (dnEi - sign * k * Er_b).pow(2)


def loss_terms(w, b, a, o, omega, circle, xy_i, xy_b, n_b):
    out_i = forward(xy_i, w, b, a, o)
    Er_i, Ei_i = out_i[:, 0], out_i[:, 1]
    eps_i = epsilon_field(xy_i, circle=circle)
    J = Jz(xy_i)
    res_r = -laplacian(Er_i, xy_i) - (eps_i * omega**2) * Er_i
    res_i = -laplacian(Ei_i, xy_i) - (eps_i * omega**2) * Ei_i + (omega * J)
    pde = (res_r.pow(2) + res_i.pow(2)).mean()
    out_b = forward(xy_b, w, b, a, o)
    bc = bc_residual(out_b[:, 0], out_b[:, 1], xy_b, n_b, omega).mean()
    return pde, bc


def measure(w, b, a, o, omega, circle):
    coords = make_plot_grid(N_EVAL).requires_grad_(True)
    out = forward(coords, w, b, a, o)
    Er, Ei = out[:, 0], out[:, 1]
    eps = epsilon_field(coords, circle=circle)
    J = Jz(coords)
    res_r = -laplacian(Er, coords) - (eps * omega**2) * Er
    res_i = -laplacian(Ei, coords) - (eps * omega**2) * Ei + (omega * J)
    pde = torch.sqrt(torch.mean(res_r**2 + res_i**2)).item()
    _, xy_b, n_b = sample_points(n_interior=100, n_boundary=8000)
    out_b = forward(xy_b, w, b, a, o)
    bc = torch.sqrt(bc_residual(out_b[:, 0], out_b[:, 1], xy_b, n_b, omega).mean()).item()
    return pde, bc


def field_of(w, b, a, o):
    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, w, b, a, o)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()
    return Er + 1j * Ei


def run(weight_file, circle, epochs, label):
    """One independent run, cosine schedule annealing over its own full length."""
    set_seed(SEED)
    w, b, a, o = load_weights(OUTDIR / weight_file)
    for t in w:
        t.requires_grad_(True)
    for t in b:
        t.requires_grad_(True)

    opt = torch.optim.Adam([*w, *b], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * ETA_MIN_FACTOR)

    for ep in range(epochs):
        opt.zero_grad()
        xy_i, xy_b, n_b = sample_points(n_interior=N_INTERIOR, n_boundary=N_BOUNDARY)
        pde, bc = loss_terms(w, b, a, o, OMEGA, circle, xy_i, xy_b, n_b)
        (pde + LAMBDA_BC * bc).backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_([*w, *b], max_norm=GRAD_CLIP)
        opt.step()
        sched.step()

    lbfgs = torch.optim.LBFGS([*w, *b], lr=1.0, max_iter=500,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        xy_i, xy_b, n_b = sample_points(n_interior=N_INTERIOR, n_boundary=N_BOUNDARY)
        pde, bc = loss_terms(w, b, a, o, OMEGA, circle, xy_i, xy_b, n_b)
        L = pde + LAMBDA_BC * bc
        L.backward()
        return L

    lbfgs.step(closure)
    pde_g, bc_g = measure(w, b, a, o, OMEGA, circle)
    print(f"  {label:<12} {epochs:>6} epochs | PDE {pde_g:.4e} | BC {bc_g:.4e}")
    return field_of(w, b, a, o), pde_g, bc_g


def rms(z):
    return float(np.sqrt(np.mean(np.abs(z) ** 2)))


def main():
    print(f"omega = {OMEGA}, corrected BC, seed = {SEED}")
    print("each budget is an INDEPENDENT run with its own annealing schedule\n")

    fields, rows = {}, []
    for n in BUDGETS:
        print(f"--- {n} epochs ---")
        fs, pf, bf = run("point_source_free_space_weights.pt", None, n, "free space")
        dl, pd_, bd = run("point_source_dielectric_sphere_weights.pt", CIRCLE, n, "dielectric")
        fields[f"free_space_{n}"] = fs
        fields[f"dielectric_{n}"] = dl
        rows.append(dict(epochs=n,
                         pde_free=pf, bc_free=bf,
                         pde_diel=pd_, bc_diel=bd,
                         rms_free=rms(fs), rms_diel=rms(dl),
                         separation=rms(fs - dl),
                         separation_relative=rms(fs - dl) / rms(dl)))

    df = pd.DataFrame(rows)

    print("\n" + "=" * 68)
    print("SEPARATION vs TRAINING LENGTH  (independent, fully annealed runs)")
    print("=" * 68)
    print(f"\n{'epochs':>8}{'PDE free':>12}{'PDE diel':>12}{'separation':>13}{'vs prev':>10}")
    prev = None
    for _, r in df.iterrows():
        chg = "--" if prev is None else f"{(r.separation/prev - 1)*100:+.1f}%"
        print(f"{int(r.epochs):>8}{r.pde_free:>12.3e}{r.pde_diel:>12.3e}"
              f"{r.separation:>13.4e}{chg:>10}")
        prev = r.separation

    print("\nfor reference, the earlier 3939-epoch run gave separation 9.561e-03")
    print("\nif the final step is small, the separation has converged and we can")
    print("state how much the dataset understates it. if it is still growing at")
    print("the same rate, it has not converged and we can only give a bound.")

    np.savez("separation_convergence_fields.npz", **fields)
    df.to_csv("separation_convergence.csv", index=False)
    print("\nwrote separation_convergence_fields.npz and separation_convergence.csv")


if __name__ == "__main__":
    main()