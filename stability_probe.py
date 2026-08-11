"""
Field-stability probe across frequency and class, using the CORRECTED
Sommerfeld condition throughout.

For each of 5 probe frequencies (4.0, 8.0, 12.0, 16.0, 19.515) and all 4
classes, one continuous run is trained through an epoch ladder
(2000, 4000, 8000, 12000, 16000, 24000), with the field recorded at each rung.

Stability criterion (fixed, identical everywhere, per Theo):
    rel_change(b_i -> b_{i+1}) = ||field(b_{i+1}) - field(b_i)|| / ||field(b_{i+1})||
A class/frequency is called STABLE at the first budget where this drops below 1%.

This measures field stability only. It does not look at the PDE residual, the
BC residual, or classification accuracy when deciding the budget -- those are
recorded for reference but play no role in the stopping decision, per Theo's
instruction that the budget should come from field stability alone.

Run on Newton:
    sbatch run_stability_probe.sh

Writes stability_probe_fields.npz and stability_probe_results.csv, and prints
a per-class, per-frequency table of the first stable budget.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path

from helmholtz_pinn import (
    forward, make_plot_grid, laplacian, epsilon_field, Jz, sample_points, device,
    LR, ETA_MIN_FACTOR, GRAD_CLIP, LAMBDA_BC, N_INTERIOR, N_BOUNDARY,
)
from helmholtz_pinn_planewave import (
    planewave_bc_loss, sample_points as sample_points_pw,
)

OUTDIR = Path("pinn_em_results")
FREQS = [4.0, 8.0, 12.0, 16.0, 19.515]
LADDER = [2000, 4000, 8000, 12000, 16000, 24000]
STABLE_THRESHOLD = 0.01          # 1% relative L2 change
SEED = 0
N_EVAL = 201
CIRCLE = (0.0, 0.0, 0.30, 2.0)
BC_SIGN = -1                     # corrected outgoing condition, dEz/dn + ikEz = 0

# (weight file, has_circle, is_planewave)
CASES = [
    ("point_source_free_space_weights.pt",        False, False, "ps_free"),
    ("point_source_dielectric_sphere_weights.pt",  True,  False, "ps_diel"),
    ("incoming_wave_weights.pt",                   False, True,  "pw_free"),
    ("incoming_wave_dielectric_sphere_weights.pt", True,  True,  "pw_diel"),
]


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


def sommerfeld_residual(Er_b, Ei_b, coords_b, normals_b, omega, sign=BC_SIGN):
    g_r = torch.autograd.grad(Er_b, coords_b, grad_outputs=torch.ones_like(Er_b),
                              create_graph=True)[0]
    g_i = torch.autograd.grad(Ei_b, coords_b, grad_outputs=torch.ones_like(Ei_b),
                              create_graph=True)[0]
    dnEr = (g_r * normals_b).sum(dim=1)
    dnEi = (g_i * normals_b).sum(dim=1)
    k = omega
    return (dnEr + sign * k * Ei_b).pow(2) + (dnEi - sign * k * Er_b).pow(2)


def loss_terms(w, b, a, o, omega, circle, is_pw, xy_i, xy_b, n_b):
    out_i = forward(xy_i, w, b, a, o)
    Er_i, Ei_i = out_i[:, 0], out_i[:, 1]
    eps_i = epsilon_field(xy_i, circle=circle)

    if is_pw:
        res_r = -laplacian(Er_i, xy_i) - (eps_i * omega**2) * Er_i
        res_i = -laplacian(Ei_i, xy_i) - (eps_i * omega**2) * Ei_i
    else:
        J = Jz(xy_i)
        res_r = -laplacian(Er_i, xy_i) - (eps_i * omega**2) * Er_i
        res_i = -laplacian(Ei_i, xy_i) - (eps_i * omega**2) * Ei_i + (omega * J)
    pde = (res_r.pow(2) + res_i.pow(2)).mean()

    out_b = forward(xy_b, w, b, a, o)
    bc_som = sommerfeld_residual(out_b[:, 0], out_b[:, 1], xy_b, n_b, omega).mean()
    if is_pw:
        bc_inlet = planewave_bc_loss(out_b[:, 0], out_b[:, 1], xy_b, omega)
        bc = bc_som + bc_inlet
    else:
        bc = bc_som
    return pde, bc


def sample_for(is_pw):
    if is_pw:
        return sample_points_pw(n_interior=N_INTERIOR, n_boundary=N_BOUNDARY)
    return sample_points(n_interior=N_INTERIOR, n_boundary=N_BOUNDARY)


def measure(w, b, a, o, omega, circle, is_pw):
    coords = make_plot_grid(N_EVAL).requires_grad_(True)
    out = forward(coords, w, b, a, o)
    Er, Ei = out[:, 0], out[:, 1]
    eps = epsilon_field(coords, circle=circle)
    if is_pw:
        res_r = -laplacian(Er, coords) - (eps * omega**2) * Er
        res_i = -laplacian(Ei, coords) - (eps * omega**2) * Ei
    else:
        J = Jz(coords)
        res_r = -laplacian(Er, coords) - (eps * omega**2) * Er
        res_i = -laplacian(Ei, coords) - (eps * omega**2) * Ei + (omega * J)
    pde = torch.sqrt(torch.mean(res_r**2 + res_i**2)).item()

    _, xy_b, n_b = sample_for(is_pw)
    out_b = forward(xy_b, w, b, a, o)
    bc = torch.sqrt(sommerfeld_residual(
        out_b[:, 0], out_b[:, 1], xy_b, n_b, omega).mean()).item()
    return pde, bc


def field_of(w, b, a, o):
    with torch.no_grad():
        coords = make_plot_grid(N_EVAL)
        out = forward(coords, w, b, a, o)
        Er = out[:, 0].reshape(N_EVAL, N_EVAL).cpu().numpy()
        Ei = out[:, 1].reshape(N_EVAL, N_EVAL).cpu().numpy()
    return Er + 1j * Ei


def rel_change(a, b):
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / np.sqrt(np.mean(np.abs(b) ** 2)))


def probe(weight_file, circle, is_pw, tag, omega):
    """One continuous run through the ladder for one class/frequency pair."""
    set_seed(SEED)
    w, b, a, o = load_weights(OUTDIR / weight_file)
    for t in w:
        t.requires_grad_(True)
    for t in b:
        t.requires_grad_(True)

    total = LADDER[-1]
    opt = torch.optim.Adam([*w, *b], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total, eta_min=LR * ETA_MIN_FACTOR)

    fields, rows = {}, []
    for ep in range(total):
        opt.zero_grad()
        xy_i, xy_b, n_b = sample_for(is_pw)
        pde, bc = loss_terms(w, b, a, o, omega, circle, is_pw, xy_i, xy_b, n_b)
        (pde + LAMBDA_BC * bc).backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_([*w, *b], max_norm=GRAD_CLIP)
        opt.step()
        sched.step()

        if (ep + 1) in LADDER:
            pde_g, bc_g = measure(w, b, a, o, omega, circle, is_pw)
            fields[ep + 1] = field_of(w, b, a, o)
            rows.append(dict(case=tag, omega=omega, budget=ep + 1,
                             pde_residual=pde_g, bc_residual=bc_g))
            print(f"    {tag:<9} omega={omega:<7} {ep+1:>6} epochs | "
                  f"PDE {pde_g:.3e} | BC {bc_g:.3e}")

    # relative change and first-stable budget
    stable_at = None
    budgets_sorted = sorted(fields.keys())
    for i in range(1, len(budgets_sorted)):
        prev_b, cur_b = budgets_sorted[i - 1], budgets_sorted[i]
        rc = rel_change(fields[cur_b], fields[prev_b])
        rows[i]["rel_change_vs_prev"] = rc
        if stable_at is None and rc < STABLE_THRESHOLD:
            stable_at = cur_b
    rows[0]["rel_change_vs_prev"] = np.nan

    for r in rows:
        r["stable_budget"] = stable_at
    print(f"    -> {tag} at omega={omega}: "
          f"{'stable at ' + str(stable_at) if stable_at else 'NOT stable by ' + str(total)}")

    return fields, rows


def main():
    print("field-stability probe, corrected BC, seed =", SEED)
    print(f"threshold: relative change < {STABLE_THRESHOLD:.0%}\n")

    all_fields, all_rows = {}, []
    for weight_file, has_circle, is_pw, tag in CASES:
        circle = CIRCLE if has_circle else None
        print(f"\n{'='*64}\n{tag}\n{'='*64}")
        for om in FREQS:
            key = f"{tag}_omega{om}"
            fields, rows = probe(weight_file, circle, is_pw, tag, om)
            for b, z in fields.items():
                all_fields[f"{key}_{b}"] = z
            all_rows += rows

    df = pd.DataFrame(all_rows)
    df.to_csv("stability_probe_results.csv", index=False)
    np.savez("stability_probe_fields.npz", **all_fields)

    print("\n" + "=" * 64)
    print("SUMMARY: first stable budget by class and frequency")
    print("=" * 64)
    summary = (df.dropna(subset=["stable_budget"])
                 .groupby(["case", "omega"])["stable_budget"].first()
                 .unstack("omega"))
    print(summary.to_string())

    print("\nCases with no entry did not stabilise within the ladder tested "
          "(up to 24000 epochs) and would need an extended ladder.")
    print("\nwrote stability_probe_results.csv and stability_probe_fields.npz")


if __name__ == "__main__":
    main()