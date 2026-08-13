"""
Extended field-stability test for three representative cases, per Theo's
follow-up instructions after the initial 20-case probe found that most
combinations had not stabilized within 24,000 epochs.

Cases tested (chosen to isolate frequency as the variable for the plane-wave
pair, since Theo did not specify free vs dielectric):
    1. plane wave, free space,     omega = 4.0     (low frequency)
    2. plane wave, free space,     omega = 19.515   (high frequency)
    3. point source, free space,   omega = 19.515   (high frequency)

Each case is trained in ONE continuous run with a single cosine schedule,
T_max = 64000, checkpointed at 2000, 4000, 8000, 12000, 16000, 24000, 32000,
and 64000. Using one schedule for all checkpoints (rather than independent
runs per budget) is intentional here: Theo's question is specifically about
the trend of successive relative changes WITHIN continued training, which is
exactly what a single run's checkpoints measure correctly. This differs from
the earlier convergence study, where checkpoints from a run scheduled for one
total length were incorrectly compared against an independently-scheduled run
of a different length.

At every checkpoint we record:
    PDE residual, BC residual (reference only, not used for the stopping
    decision)
    relative L2 change vs the previous checkpoint
    relative L2 change specifically for 24k->32k and 32k->64k, the two
    Theo asked to see directly

The question is whether relative change keeps shrinking toward the 1%
threshold, or levels off above it. Both are recorded explicitly at the end.

Run on Newton:
    sbatch run_extended_stability.sh

Writes extended_stability_fields.npz and extended_stability_results.csv.
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
CHECKPOINTS = [2000, 4000, 8000, 12000, 16000, 24000, 32000, 64000]
STABLE_THRESHOLD = 0.01
SEED = 0
N_EVAL = 201
BC_SIGN = -1  # corrected outgoing condition, dEz/dn + ikEz = 0

# (tag, weight_file, has_circle, is_pw, omega)
CASES = [
    ("pw_free_low",  "incoming_wave_weights.pt", False, True,  4.0),
    ("pw_free_high", "incoming_wave_weights.pt", False, True,  19.515),
    ("ps_free_high", "point_source_free_space_weights.pt", False, False, 19.515),
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


def run_case(tag, weight_file, circle, is_pw, omega):
    print(f"\n{'='*64}\n{tag}  (omega={omega})\n{'='*64}")
    set_seed(SEED)
    w, b, a, o = load_weights(OUTDIR / weight_file)
    for t in w:
        t.requires_grad_(True)
    for t in b:
        t.requires_grad_(True)

    total = CHECKPOINTS[-1]
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

        if (ep + 1) in CHECKPOINTS:
            pde_g, bc_g = measure(w, b, a, o, omega, circle, is_pw)
            fields[ep + 1] = field_of(w, b, a, o)
            print(f"    {ep+1:>6} epochs | PDE {pde_g:.4e} | BC {bc_g:.4e}")
            rows.append(dict(case=tag, omega=omega, budget=ep + 1,
                             pde_residual=pde_g, bc_residual=bc_g))

    print(f"\n  relative L2 change between successive checkpoints:")
    cps = sorted(fields.keys())
    for i in range(1, len(cps)):
        prev_b, cur_b = cps[i - 1], cps[i]
        rc = rel_change(fields[cur_b], fields[prev_b])
        rows[i]["rel_change_vs_prev"] = rc
        flag = "  <-- Theo asked to see this one" if (prev_b, cur_b) in \
            [(24000, 32000), (32000, 64000)] else ""
        print(f"    {prev_b:>6} -> {cur_b:>6}: {rc:.4%}{flag}")
    rows[0]["rel_change_vs_prev"] = np.nan

    return fields, rows


def main():
    print("extended stability test, corrected BC, seed =", SEED)
    print(f"checkpoints: {CHECKPOINTS}")
    print("cases:")
    for tag, wf, circ, pw, om in CASES:
        print(f"  {tag}: omega={om}, {'plane wave' if pw else 'point source'}, "
              f"{'dielectric' if circ else 'free space'}")

    all_fields, all_rows = {}, []
    for tag, wf, has_circ, is_pw, om in CASES:
        circle = None  # all three cases are free space per Theo's selection
        fields, rows = run_case(tag, wf, circle, is_pw, om)
        for b, z in fields.items():
            all_fields[f"{tag}_{b}"] = z
        all_rows += rows

    df = pd.DataFrame(all_rows)
    df.to_csv("extended_stability_results.csv", index=False)
    np.savez("extended_stability_fields.npz", **all_fields)

    print("\n" + "=" * 64)
    print("TREND: is relative change still decreasing toward 1%, or leveling off?")
    print("=" * 64)
    for tag, wf, circ, pw, om in CASES:
        sub = df[df.case == tag].sort_values("budget")
        print(f"\n{tag}  (omega={om})")
        vals = sub[["budget", "rel_change_vs_prev"]].dropna()
        for _, r in vals.iterrows():
            marker = " *** below 1% ***" if r.rel_change_vs_prev < STABLE_THRESHOLD else ""
            print(f"    at {int(r.budget):>6}: {r.rel_change_vs_prev:.4%}{marker}")
        last3 = vals.rel_change_vs_prev.tail(3).values
        if len(last3) >= 2:
            trend = "decreasing" if last3[-1] < last3[-2] else "flat or increasing"
            print(f"    trend over final steps: {trend}")

    print("\nwrote extended_stability_results.csv and extended_stability_fields.npz")


if __name__ == "__main__":
    main()