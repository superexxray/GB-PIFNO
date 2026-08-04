# GB-PIFNO: Gradient-Balanced Physics-Informed Fourier Neural Operator

> Robust spectral surrogate modeling for turbulent fluid dynamics under extreme data sparsity.

GB-PIFNO is a Scientific Machine Learning (SciML) research project that investigates robust neural operator learning for **2D incompressible Navier–Stokes equations** using adaptive physics-informed optimization. The project extends the **Fourier Neural Operator (FNO)** with dynamic gradient balancing to improve training stability when learning from sparse and noisy observations.

The repository provides implementations of standard data-driven FNOs, physics-informed FNOs, sparse sensor simulation, spectral evaluation metrics, and training pipelines for benchmarking operator learning methods.

---

## Motivation

Traditional Computational Fluid Dynamics (CFD) simulations are computationally expensive. Neural Operators such as the Fourier Neural Operator (FNO) provide fast surrogate models but often fail when:

- Training data is extremely sparse
- Sensor observations are noisy
- Fine-scale turbulent structures must be reconstructed
- Physical consistency is required

Physics-Informed FNOs introduce PDE residual losses but frequently suffer from **gradient imbalance**, where the physics loss dominates optimization and destabilizes training.

GB-PIFNO addresses this issue through **dynamic gradient-based loss balancing**.

---

## Project Goals

- Learn surrogate models for turbulent fluid flows
- Improve robustness under sparse observations
- Incorporate Navier–Stokes physics during training
- Balance supervised and PDE losses dynamically
- Preserve physically meaningful turbulent structures
- Evaluate both numerical accuracy and spectral fidelity

---

## Methodology

The project compares multiple approaches:

- **U-Net** — Data-driven convolutional baseline
- **FNO** — Fourier Neural Operator
- **Static PIFNO** — Fixed-weight Physics-Informed FNO
- **GB-PIFNO** — Dynamic Gradient-Balanced Physics-Informed FNO

GB-PIFNO dynamically adjusts the physics-loss weight during training using the relative parameter-gradient magnitudes of the supervised and PDE losses.

Spatial derivatives are computed directly in the Fourier domain, enabling accurate spectral PDE residual evaluation.

---

## Features

- Fourier Neural Operator (FNO)
- U-Net baseline
- Static Physics-Informed FNO
- Gradient-Balanced Physics-Informed FNO
- Fourier-domain derivative operators
- Sparse sensor simulation
- Relative L₂ evaluation
- Sobolev (H¹) error evaluation
- Radial energy spectrum computation
- Enstrophy computation
- Gradient norm diagnostics
- Training and evaluation pipelines

---

## Repository Structure

```text
GB-PIFNO/
│
├── dataset.py                # Dataset loading
├── fno.py                    # Fourier Neural Operator
├── unet.py                   # U-Net baseline
├── deeponet.py               # DeepONet baseline
├── pde_residual.py           # Navier–Stokes residual
├── train.py                  # Data-only training
├── train_static_pifno.py     # Static Physics-Informed training
├── train_gb_pifno.py         # GB-PIFNO training
├── eval_metrics.py           # Evaluation metrics
├── sparsity.py               # Sparse sensor simulation
├── plot_diagnostics.py       # Visualization utilities
└── README.md
```

---

## Evaluation Metrics

Model performance is evaluated using:

- Relative L₂ Error
- Relative H¹ (Sobolev) Error
- PDE Residual Norm
- Enstrophy
- Radial Kinetic Energy Spectrum
- Gradient Norm Diagnostics

---

## Dataset

The dataset is **not included** in this repository because of GitHub file size limits.

Download the Navier–Stokes dataset separately and place it in the appropriate directory before training.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/superexxray/GB-PIFNO.git
cd GB-PIFNO
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

Train the data-driven FNO:

```bash
python train.py
```

Train Static PIFNO:

```bash
python train_static_pifno.py
```

Train GB-PIFNO:

```bash
python train_gb_pifno.py
```

---

## Research Status

This repository is an active research project. Features, experiments, and documentation are under continuous development.

Current development includes:

- Physics-informed training
- Dynamic gradient balancing
- Sparse observation experiments
- Turbulence spectrum analysis
- Robustness benchmarking

---

## Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- Matplotlib

---

## Citation

If you use this repository in your research, please cite the associated paper (coming soon).

---

## License

This project is released for research and educational purposes.
