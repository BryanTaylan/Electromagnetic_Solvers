# Electromagnetic Wave Solvers

This repository provides two complementary solvers for electromagnetic wave propagation:
1. **PINN-based Helmholtz Solver** using **SIREN networks** (`helmholtz_pinn.py`)
2. **Classical FDFD Eigenmode Solver** for silicon photonic waveguides (`waveguide_fdfd.py`)

Both solve Maxwell’s equations in the frequency domain but with very different numerical philosophies:
- The **PINN** learns the solution directly from the governing PDE and boundary conditions.
- The **FDFD** uses finite-difference discretization and sparse-matrix eigensolvers.

---

##  Project Structure

```
electromagnetic-wave-solvers/
│
├── helmholtz_pinn.py      # Physics-Informed Neural Network (SIREN-based Helmholtz solver)
├── waveguide_fdfd.py      # 2D vector FDFD mode solver for rectangular Si waveguides              
├── pinn_em_results/       
├── README.md
└── requirements.txt
```

---

##  Requirements

Install required dependencies:
```bash
pip install torch numpy scipy matplotlib
```

For GPU acceleration (PINN training), install CUDA-enabled PyTorch.

---

##  Usage

###  1. Physics-Informed Neural Network (PINN)

Solves the **2D Helmholtz equation** with a SIREN neural network.

```bash
python helmholtz_pinn.py
```

**Key features:**
- Helmholtz PDE residual + Sommerfeld boundary condition loss  
- Dual-optimizer training: **Adam + L-BFGS**  
- Visualization of real/imaginary E-fields  
- Support for dielectric inclusions  
- Computes RMS PDE residual  

**Outputs:**
- `pinn_em_results/free_space_omega8.png`
- `pinn_em_results/dielectric_circle_omega8.png`
- Printed PDE residual RMS

---

###  2. Finite-Difference Frequency-Domain (FDFD) Waveguide Solver

Computes **guided TE/TM modes** of a silicon wire waveguide and tracks dispersion over wavelength.

```bash
python waveguide_fdfd.py
```

**Capabilities:**
- Vector full-field Maxwell eigenvalue formulation  
- **Anti-aliased permittivity fill** (subpixel accuracy)  
- **Shift-invert eigensolver** for targeted TE/TM clusters  
- **Mode tracking** by overlap through a wavelength sweep  
- Field reconstruction (E, H components)  
- Propagation overlap diagnostics  

**Outputs:**
- Mode field plots: `|Ex|`, `|Ey|`, `|Ez|`, and total `|E|`
- Dispersion curves: `n_eff(λ)` for multiple branches
- Self-overlap vs propagation distance

---

##  Comparison

| Feature                     | `helmholtz_pinn.py` (PINN) | `waveguide_fdfd.py` (FDFD) |
|------------------------------|-----------------------------|-----------------------------|
| Solver type                  | Neural PDE (PINN)           | Finite-difference eigenmode |
| Equation solved              | Helmholtz (2D EM)           | Maxwell curl equations (full vector) |
| Boundary condition           | Sommerfeld (radiation)      | Dirichlet (metal walls)     |
| Outputs                      | E-field maps (Re/Im)        | Guided mode profiles, dispersion |
| Training/solve method        | Adam + L-BFGS               | Sparse eigenvalue (`eigsh`) |
| Primary goal                 | PINN accuracy/stability test| Modal analysis of waveguides |

---

##  Example Results

**PINN Output (Helmholtz):**
- Real and imaginary parts of \(E_z\)  
- Source and permittivity distribution  

**FDFD Output (Waveguide):**
- Fundamental TE/TM mode field profiles  
- Effective index dispersion \(n_\mathrm{eff}(\lambda)\)  
- Mode self-overlap after propagation  

---


---

##  License

MIT License © 2025

---

##  References

- **SIREN** — Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions*, NeurIPS 2020  
- **PINNs** — Raissi et al., *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear PDEs*, JCP 2019  
- **FDFD Theory** — Sadiku, *Numerical Techniques in Electromagnetics*, 3rd Edition
