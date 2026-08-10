"""
PDE residual across the frequency sweep, for every generated dataset sample,
reported separately for the four physical classes.

The stored samples are field arrays rather than networks, so the Laplacian is
taken by finite differences on the 201 x 201 grid rather than by automatic
differentiation. A fourth-order stencil is used; the second-order value is also
computed and reported as a discretisation-error check. At omega = 20 the
estimated relative error in the Laplacian term is about 2e-5 for fourth order
against 3e-3 for second order, so the fourth-order figure is the one to quote.

Points within two cells of the boundary are excluded, since the stencil does not
fit there. The residual is therefore an interior measure and does not include
boundary behaviour, which is reported separately by the convergence study.

Sign convention matches the implementation:
    res_r = -lap_r - eps*omega^2*Er
    res_i = -lap_i - eps*omega^2*Ei + omega*Jz

Run on Newton:
    sbatch run_residual_sweep.sh

Writes residual_vs_frequency.csv and prints a summary by class.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATASET = Path("dataset")
METADATA = DATASET / "metadata.csv"
N = 201
H = 2.0 / (N - 1)
J_SIGMA = 0.05
CIRCLE_R = 0.30
EPS_IN = 2.0

# class label -> (has dielectric inclusion, has Gaussian source)
CLASS_INFO = {
    0: (False, True,  "point source, free space"),
    1: (True,  True,  "point source, dielectric"),
    2: (False, False, "plane wave, free space"),
    3: (True,  False, "plane wave, dielectric"),
}


def laplacian_2nd(u, h):
    """Second-order five-point Laplacian. Returns the valid interior."""
    return (u[:-2, 1:-1] + u[2:, 1:-1] +
            u[1:-1, :-2] + u[1:-1, 2:] -
            4.0 * u[1:-1, 1:-1]) / h**2


def laplacian_4th(u, h):
    """Fourth-order Laplacian, valid two cells in from each edge."""
    c = u[2:-2, 2:-2]
    # d2/dx2 along axis 1
    d2x = (-u[2:-2, 4:] + 16*u[2:-2, 3:-1] - 30*c
           + 16*u[2:-2, 1:-3] - u[2:-2, :-4]) / (12*h**2)
    # d2/dy2 along axis 0
    d2y = (-u[4:, 2:-2] + 16*u[3:-1, 2:-2] - 30*c
           + 16*u[1:-3, 2:-2] - u[:-4, 2:-2]) / (12*h**2)
    return d2x + d2y


def grids():
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2
    Jz = np.exp(-R2 / (2.0 * J_SIGMA**2))
    return X, Y, R2, Jz


X, Y, R2, JZ = grids()
INSIDE = R2 <= CIRCLE_R**2


def residual_rms(Er, Ei, omega, has_dielectric, has_source, order=4):
    eps = np.ones((N, N))
    if has_dielectric:
        eps[INSIDE] = EPS_IN
    J = JZ if has_source else np.zeros((N, N))

    if order == 4:
        lap_r = laplacian_4th(Er, H)
        lap_i = laplacian_4th(Ei, H)
        sl = (slice(2, -2), slice(2, -2))
    else:
        lap_r = laplacian_2nd(Er, H)
        lap_i = laplacian_2nd(Ei, H)
        sl = (slice(1, -1), slice(1, -1))

    res_r = -lap_r - eps[sl] * omega**2 * Er[sl]
    res_i = -lap_i - eps[sl] * omega**2 * Ei[sl] + omega * J[sl]
    return float(np.sqrt(np.mean(res_r**2 + res_i**2)))


def main():
    meta = pd.read_csv(METADATA)
    print(f"loaded {len(meta)} samples from {METADATA}\n")

    rows = []
    for _, r in meta.iterrows():
        arr = np.load(r["filepath"])
        Er, Ei = arr[0], arr[1]
        lbl = int(r["class_label"])
        has_d, has_s, name = CLASS_INFO[lbl]
        om = float(r["frequency"])

        r4 = residual_rms(Er, Ei, om, has_d, has_s, order=4)
        r2 = residual_rms(Er, Ei, om, has_d, has_s, order=2)

        rows.append(dict(
            filepath=r["filepath"], class_label=lbl, class_name=name,
            frequency=om,
            residual_rms=r4,
            residual_rms_2nd_order=r2,
            fd_discrepancy=abs(r4 - r2) / r4 if r4 > 0 else np.nan,
            field_rms=float(np.sqrt(np.mean(Er**2 + Ei**2))),
        ))

    df = pd.DataFrame(rows).sort_values(["class_label", "frequency"])
    df.to_csv("residual_vs_frequency.csv", index=False)

    print("=" * 74)
    print("PDE RESIDUAL BY CLASS AND FREQUENCY BAND")
    print("=" * 74)
    bands = [(4, 8), (8, 12), (12, 16), (16, 20.1)]
    for lbl, (_, _, name) in CLASS_INFO.items():
        sub = df[df.class_label == lbl]
        print(f"\n{name}")
        print(f"  {'band':<14}{'n':>4}{'median':>12}{'min':>12}{'max':>12}")
        for lo, hi in bands:
            b = sub[(sub.frequency >= lo) & (sub.frequency < hi)]
            if not len(b):
                continue
            print(f"  omega {lo:>2}-{hi:<6.0f}{len(b):>4}"
                  f"{b.residual_rms.median():>12.3e}"
                  f"{b.residual_rms.min():>12.3e}"
                  f"{b.residual_rms.max():>12.3e}")

    print("\n" + "=" * 74)
    print("TEST-SET BAND ONLY (omega >= 13.9)")
    print("=" * 74)
    hi = df[df.frequency >= 13.9]
    for lbl, (_, _, name) in CLASS_INFO.items():
        s = hi[hi.class_label == lbl]
        print(f"  {name:<28} median {s.residual_rms.median():.3e}  "
              f"n = {len(s)}")

    print(f"\nfinite-difference check: median |4th - 2nd| / 4th = "
          f"{df.fd_discrepancy.median():.2%}")
    print(f"  worst case {df.fd_discrepancy.max():.2%} at omega = "
          f"{df.loc[df.fd_discrepancy.idxmax(), 'frequency']:.2f}")

    print("\nwrote residual_vs_frequency.csv")


if __name__ == "__main__":
    main()