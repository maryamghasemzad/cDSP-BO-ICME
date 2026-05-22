# cDSP-BO-ICME

**Bayesian Optimization integrated with the compromise Decision Support Problem (cDSP) for robust, uncertainty-aware design in Integrated Computational Materials Engineering (ICME).**

This repository contains the reference implementation for the mathematical example in:

> H. M. Dilshad Alam Digonta, Maryam Ghasemzadeh, Anton van Beek, and Anand Balu Nellippallil.
> *Design for ICME — A Data-Driven Decision Support Framework for Quantifying and Managing Uncertainty.*

A Gaussian Process surrogate is fitted via Maximum-A-Posteriori (MAP) hyperparameter estimation, and a cDSP-based Expected Improvement (EI) acquisition function steers sampling toward **robust satisficing** regions of the design space rather than the unconstrained global optimum.

---

## Overview

Realising ICME from a design perspective involves co-considering the interactions between manufacturing, materials, and product disciplines. Each discipline introduces its own sources of uncertainty (experimental variability, model assumptions, limited data) that propagate through the process chain. The framework in this repository

- builds a **Gaussian Process surrogate** that quantifies predictive uncertainty,
- uses **Bayesian Optimization** to adaptively acquire information-efficient new samples, and
- formulates design goals through the **cDSP construct** with the **Error Margin Index (EMI)** so the acquisition function looks for solutions that are *relatively insensitive* to variability.

The toy problem used here is a 2-D quadratic interaction \( f(x_1,x_2) = (x_1+x_2)^2 \) over the search domain \([0,100]^2\).

---

## Repository layout

```
cDSP-BO-ICME/
├── main.py                                # All algorithmic code (GP, MAP, EMI, EI, BO loop)
├── notebooks/
│   └── Single_objective_BO_cDSP.ipynb     # Cleaned tutorial notebook with explanations
├── figures/                               # Plots written by main.py
├── requirements.txt                       # Pinned Python dependencies
├── LICENSE                                # MIT
├── CITATION.cff                           # Citation metadata for GitHub's "Cite this repository"
└── README.md
```

---

## Installation

Tested with Python 3.10+. We recommend a virtual environment:

```bash
git clone https://github.com/<your-username>/cDSP-BO-ICME.git
cd cDSP-BO-ICME

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Run the example end-to-end

```bash
python main.py
```

This executes 100 iterations of the cDSP-based BO loop on the toy objective and writes two figures to `figures/`:

- `figures/samples.png` — initial LHS design vs. BO-acquired points.
- `figures/convergence.png` — best-so-far convergence trace.

### Use the notebook

```bash
jupyter notebook notebooks/Single_objective_BO_cDSP.ipynb
```

The notebook walks through every step (problem set-up → GP → MAP → EMI/cDSP → EI → BO loop → plots) with markdown explanations matching the paper.

### Use the library in your own code

```python
from main import run_optimisation, plot_samples, plot_convergence

X, Y = run_optimisation(n_iterations=50, target=10.0, emi_target=10.0)
plot_samples(X, n_initial=10, save_path="samples.png")
plot_convergence(Y, save_path="convergence.png")
```

To apply the framework to your own problem, replace `true_function` in `main.py` with your simulator/experiment evaluator and adjust `build_test_grid` / `initial_design` to your design space.

---

## Methodology

**1. Gaussian Process surrogate.** A zero-mean GP with a squared-exponential kernel
\[
k(x_i,x_j) = \exp\!\big(-10^{\omega_0}\,\|x_i-x_j\|^2\big),
\]
parameterised by `omega = [log10 roughness, log10 nugget]`.

**2. MAP hyperparameter estimation.** Multi-start L-BFGS-B over Latin Hypercube initial points maximises the posterior \(p(\omega\mid X,y)\propto p(y\mid X,\omega)\,p(\omega)\). Priors: standard normal on roughness, log-normal on nugget.

**3. Error Margin Index (EMI).** For a larger-is-better convention,
\[
\mathrm{EMI}(\mu,t,y_{\min}) = \frac{\mu - t}{\mu - y_{\min}}.
\]

**4. cDSP deviation.** \(d = 1 - \mathrm{EMI}/\mathrm{EMI}_{\text{target}}\) — minimising \(d\) drives \(\mathrm{EMI}\) toward its target from below.

**5. Acquisition.** Expected Improvement is computed on \(d\) (not on the raw response), so the BO loop searches for *robust* points.

---

## Changes from the original Colab notebook

The original notebook was preserved logically but cleaned up:

- Refactored a single Colab notebook into a documented Python module (`main.py`) plus a clean tutorial notebook.
- Switched the hyperparameter optimiser from `BFGS` (which silently ignores bounds) to `L-BFGS-B`, removing the `Method BFGS cannot handle bounds` warning.
- Used `np.linalg.slogdet` for numerically stable log-determinants, eliminating the `divide by zero encountered in log` warning when the kernel determinant underflows.
- Fixed a bug in `expected_improvement` where the *objective values* of the multi-start optimiser were being passed to the GP as `omega` instead of the optimiser's argmin.
- Removed Colab-only imports (`google.colab.files`) and seeded the RNG for reproducibility.
- Added type hints and docstrings throughout.

---

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{digonta2025icme,
  title   = {Design for ICME -- A Data-Driven Decision Support Framework for
             Quantifying and Managing Uncertainty},
  author  = {Digonta, H. M. Dilshad Alam and Ghasemzadeh, Maryam and
             van Beek, Anton and Nellippallil, Anand Balu},
  journal = {Integrating Materials and Manufacturing Innovation},
  year    = {2026}
}
```

A `CITATION.cff` file is included so GitHub will surface a "Cite this repository" button automatically.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Contact

- Maryam Ghasemzadeh — University College Dublin (UCD)
  

Issues and pull requests are welcome.
