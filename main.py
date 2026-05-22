"""
Single-objective Bayesian Optimization integrated with the compromise
Decision Support Problem (cDSP) construct.

This script reproduces the mathematical example reported in:

    H. M. Dilshad Alam Digonta, Maryam Ghasemzadeh, Anton van Beek, and
    Anand Balu Nellippallil, "Design for ICME - A Data-Driven Decision
    Support Framework for Quantifying and Managing Uncertainty."

The framework couples a Gaussian Process (GP) surrogate with a cDSP-based
acquisition strategy. The Error Margin Index (EMI) is used inside the
deviation function `d` so that the acquisition function (Expected
Improvement) directs sampling toward *robust satisficing* regions of the
design space rather than the global optimum alone.

Pipeline at every BO iteration
------------------------------
1. Generate Latin Hypercube samples of GP hyperparameters (omega = [log10
   roughness, log10 nugget]) and find the Maximum-A-Posteriori (MAP)
   hyperparameters by maximising the unnormalised posterior.
2. Predict mean and variance of the response on a test grid with the
   fitted GP.
3. Convert the posterior mean to the Error Margin Index (EMI) and then
   to a deviation `d` (cDSP goal).
4. Use Expected Improvement on `d` to pick the next sampling point.
5. Evaluate the true function at that point, append to the dataset.

Run directly with::

    python main.py

Outputs (PNG plots) are written next to the script.
"""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# 1.  Gaussian Process building blocks
# ---------------------------------------------------------------------------
def kernel(x1: np.ndarray, x2: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Squared-exponential (RBF) covariance kernel.

    Parameters
    ----------
    x1, x2 : ndarray, shape (n, d) and (m, d)
        Input matrices.
    omega : ndarray, shape (2,)
        Hyperparameters in log10-space: ``omega[0]`` is the inverse length
        scale (roughness) and ``omega[1]`` is the nugget (only used in
        ``gp_regression``; included here for a uniform signature).

    Returns
    -------
    K : ndarray, shape (n, m)
        Covariance matrix ``K_ij = exp(-10**omega[0] * ||x1_i - x2_j||^2)``.
    """
    sqdist = (
        np.sum(x1 ** 2, axis=1).reshape(-1, 1)
        + np.sum(x2 ** 2, axis=1)
        - 2 * np.dot(x1, x2.T)
    )
    return np.exp(-(10 ** omega[0]) * sqdist)


def _safe_kernel_and_inverse(
    X: np.ndarray, omega: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``K + nugget * I`` and its inverse, guaranteeing positivity.

    If the kernel matrix is numerically singular or has non-positive
    determinant, the nugget is increased to ``|min_eigenvalue| + 1e-8``.
    """
    nugget = 10 ** omega[1]
    K = kernel(X, X, omega) + nugget * np.eye(X.shape[0])

    if np.linalg.det(K) <= 0:
        eigenvalues = np.linalg.eigvals(K)
        nugget = 1e-8 - float(np.real(np.min(eigenvalues)))
        K = kernel(X, X, omega) + nugget * np.eye(X.shape[0])

    return K, np.linalg.inv(K)


def gp_regression(
    X: np.ndarray, y: np.ndarray, X_star: np.ndarray, omega: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean and covariance of a zero-mean GP.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Training inputs.
    y : ndarray, shape (n, 1)
        Training targets.
    X_star : ndarray, shape (m, d)
        Test inputs at which to predict.
    omega : ndarray, shape (2,)
        Kernel hyperparameters (see :func:`kernel`).

    Returns
    -------
    f_star : ndarray, shape (m, 1)
        Posterior mean.
    var_f_star : ndarray, shape (m, m)
        Posterior covariance scaled by the MLE process variance.
    """
    n = len(X)
    K, K_inv = _safe_kernel_and_inverse(X, omega)

    # Maximum-likelihood estimate of the process variance.
    var = float(y.T @ K_inv @ y) / n
    if var <= 0:
        # Fall back to a strictly positive estimate.
        eigenvalues = np.linalg.eigvals(K)
        nugget = 1e-8 - float(np.real(np.min(eigenvalues)))
        K = kernel(X, X, omega) + nugget * np.eye(X.shape[0])
        K_inv = np.linalg.inv(K)
        var = float(y.T @ K_inv @ y) / n

    K_star = kernel(X, X_star, omega)
    f_star = K_star.T @ K_inv @ y
    var_f_star = (kernel(X_star, X_star, omega) - K_star.T @ K_inv @ K_star) * var
    return f_star, var_f_star


# ---------------------------------------------------------------------------
# 2.  Hyperparameter inference (MAP)
# ---------------------------------------------------------------------------
def neg_log_likelihood(
    omega: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """Negative concentrated log-likelihood of the GP.

    The process variance is profiled out (MLE).  Returned with a leading
    minus sign so that ``scipy.optimize.minimize`` can be used directly.
    """
    n = len(X)
    K, K_inv = _safe_kernel_and_inverse(X, omega)

    var = float(y.T @ K_inv @ y) / n
    if var <= 0:
        eigenvalues = np.linalg.eigvals(K)
        nugget = 1e-8 - float(np.real(np.min(eigenvalues)))
        K = kernel(X, X, omega) + nugget * np.eye(X.shape[0])
        K_inv = np.linalg.inv(K)
        var = float(y.T @ K_inv @ y) / n

    # Use slogdet for numerical stability.
    sign, logdet = np.linalg.slogdet(K)
    log_likelihood = -(n / 2) * np.log(max(var, 1e-300)) - 0.5 * logdet
    return -log_likelihood


def prior(
    roughness: float, nugget: float, rough_mean: float, nug_mean: float
) -> float:
    """Joint prior on the kernel hyperparameters.

    The roughness uses a standard normal centred on ``rough_mean``; the
    nugget uses a log-normal centred (in log10-space) on ``nug_mean``.
    """
    prior_rough = (1.0 / np.sqrt(2 * np.pi)) * np.exp(
        -((roughness - rough_mean) ** 2) / 2
    )
    prior_nugget = (1.0 / (nugget * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(nugget) - nug_mean) ** 2)
    )
    return prior_rough * prior_nugget


def neg_unnormalised_posterior(
    omega: np.ndarray,
    mean_rough: float,
    mean_nug: float,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Negative of  exp(log_likelihood) * prior  (for minimisation)."""
    log_lik = -neg_log_likelihood(omega, X, y)
    prior_value = prior(omega[0], 10 ** omega[1], mean_rough, mean_nug)
    posterior = np.exp(log_lik) * prior_value

    if not np.isfinite(posterior):
        return 1.0  # treat invalid samples as bad (we minimise the negative)
    return -float(posterior)


def fit_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_starts: int = 10,
    rough_mean: float = -2.5,
    nug_mean: float = 10 ** -3.5,
    bounds: Tuple[Tuple[float, float], ...] = ((-5, 0), (-8, 1)),
) -> np.ndarray:
    """MAP estimate of the GP hyperparameters via multi-start L-BFGS-B.

    Parameters
    ----------
    X, y : ndarray
        Training inputs and targets.
    n_starts : int
        Number of Latin Hypercube starting points.
    rough_mean, nug_mean : float
        Means used in :func:`prior`.
    bounds : tuple of (low, high)
        Box constraints on ``omega = [log10 roughness, log10 nugget]``.

    Returns
    -------
    omega_best : ndarray, shape (2,)
        Hyperparameters with the largest unnormalised posterior.
    """
    lhs = qmc.LatinHypercube(d=2).random(n=n_starts)
    starts = qmc.scale(
        lhs,
        l_bounds=[bounds[0][0] + 0.2, bounds[1][0] + 0.2],
        u_bounds=[bounds[0][1] - 0.2, bounds[1][1] - 0.2],
    )

    best_fun = np.inf
    best_omega = starts[0]
    for omega0 in starts:
        # NOTE: BFGS does *not* accept bounds; using L-BFGS-B fixes the
        # `Method BFGS cannot handle bounds` warning from the original code.
        result = minimize(
            neg_unnormalised_posterior,
            omega0,
            args=(rough_mean, nug_mean, X, y),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_fun:
            best_fun = result.fun
            best_omega = result.x
    return np.asarray(best_omega)


# ---------------------------------------------------------------------------
# 3.  cDSP / EMI definitions
# ---------------------------------------------------------------------------
def emi(mu: np.ndarray, target: float, y_min: float) -> np.ndarray:
    """Error Margin Index, *larger-is-better* convention.

    ``EMI = (mu - target) / (mu - y_min)``
    """
    return (mu - target) / (mu - y_min)


def deviation(emi_value: np.ndarray, emi_target: float) -> np.ndarray:
    """cDSP deviation variable: ``d = 1 - emi / emi_target``.

    Minimising ``d`` drives ``emi`` toward ``emi_target`` from below.
    """
    return 1.0 - (emi_value / emi_target)


# ---------------------------------------------------------------------------
# 4.  Acquisition function (Expected Improvement on the cDSP goal)
# ---------------------------------------------------------------------------
def expected_improvement(
    X: np.ndarray,
    y: np.ndarray,
    X_star: np.ndarray,
    target: float,
    emi_target: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function combined with cDSP.

    The GP is refit at every call (re-running hyperparameter MAP), then
    Expected Improvement is computed on the cDSP deviation ``d`` so that
    the BO loop searches for *robust* points rather than only minima of
    the raw response.

    Parameters
    ----------
    X, y : ndarray
        Observed inputs and outputs so far.
    X_star : ndarray
        Candidate test grid.
    target : float
        Target value used inside :func:`emi`.
    emi_target : float
        Target value used inside :func:`deviation`.
    xi : float
        Exploration parameter (small positive shift).

    Returns
    -------
    ei : ndarray, shape (m, 1)
        Expected Improvement at every candidate point.
    """
    omega = fit_hyperparameters(X, y)

    mu, sigma_full = gp_regression(X, y, X_star, omega)
    sigma = np.maximum(sigma_full.diagonal().reshape(-1, 1), 0.0)  # variance

    # Sample-based estimate of the per-point minimum used inside EMI.
    rng = np.random.default_rng(RNG_SEED)
    y_min = np.empty(len(X_star))
    for i in range(len(X_star)):
        draws = rng.normal(float(mu[i]), float(np.sqrt(sigma[i])), size=1000)
        y_min[i] = np.min(draws)
    y_min = np.min(y_min)

    emi_value = emi(mu, target, y_min)
    d_value = deviation(emi_value, emi_target)
    best_value = float(np.min(d_value))

    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = best_value - mu - xi
        sigma_std = np.sqrt(sigma)
        Z = np.where(sigma_std > 0, improvement / sigma_std, 0.0)
        ei = improvement * norm.cdf(Z) + sigma_std * norm.pdf(Z)
        ei[sigma_std == 0.0] = 0.0
    return ei


# ---------------------------------------------------------------------------
# 5.  Toy objective and optimisation loop
# ---------------------------------------------------------------------------
def true_function(x: np.ndarray) -> np.ndarray:
    """Quadratic interaction toy function from the paper.

    ``f(x1, x2) = x1^2 + 2 x1 x2 + x2^2 = (x1 + x2)^2``
    """
    x = np.atleast_2d(x)
    return x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1] + x[:, 1] ** 2


def build_test_grid(
    lower: float = 0.0, upper: float = 100.0, n: int = 100
) -> np.ndarray:
    """Uniform 2-D grid of ``n x n`` candidate points in ``[lower, upper]^2``."""
    axis = np.linspace(lower, upper, n)
    X1, X2 = np.meshgrid(axis, axis)
    return np.column_stack([X1.ravel(), X2.ravel()])


def initial_design(
    n_samples: int = 10, lower: float = 0.0, upper: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Latin Hypercube initial design in ``[lower, upper]^2``."""
    sampler = qmc.LatinHypercube(d=2, seed=RNG_SEED)
    x = qmc.scale(sampler.random(n=n_samples), l_bounds=[lower, lower],
                  u_bounds=[upper, upper])
    y = true_function(x).reshape(-1, 1)
    return x, y


def run_optimisation(
    n_iterations: int = 100,
    target: float = 10.0,
    emi_target: float = 10.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full cDSP-based Bayesian Optimisation loop."""
    x_sampled, y_sampled = initial_design()
    x_test = build_test_grid()

    for i in range(n_iterations):
        ei = expected_improvement(
            x_sampled, y_sampled, x_test, target, emi_target, xi=0.01
        )
        x_index = int(np.argmax(ei))
        x_new = x_test[x_index]
        y_new = float(true_function(x_new))

        x_sampled = np.append(x_sampled, x_new.reshape(1, -1), axis=0)
        y_sampled = np.append(
            y_sampled, np.array(y_new).reshape(1, 1), axis=0
        )
        if verbose and (i + 1) % 10 == 0:
            print(
                f"  iter {i + 1:3d}/{n_iterations}: "
                f"x_new={x_new}, y_new={y_new:.3f}"
            )

    return x_sampled, y_sampled


# ---------------------------------------------------------------------------
# 6.  Visualisation
# ---------------------------------------------------------------------------
def plot_samples(
    x_sampled: np.ndarray,
    n_initial: int = 10,
    save_path: str | None = None,
) -> None:
    """Scatter plot of initial vs BO-acquired samples."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        x_sampled[:n_initial, 0],
        x_sampled[:n_initial, 1],
        s=60,
        marker="o",
        edgecolor="black",
        facecolor="tab:blue",
        label="Initial LHS",
    )
    ax.scatter(
        x_sampled[n_initial:, 0],
        x_sampled[n_initial:, 1],
        s=30,
        marker="x",
        color="tab:red",
        label="BO-acquired",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Sample distribution after cDSP-BO")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.close(fig)


def plot_convergence(y_sampled: np.ndarray, save_path: str | None = None) -> None:
    """Best-so-far convergence trace."""
    best_so_far = np.minimum.accumulate(y_sampled.ravel())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best_so_far, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best observed $y$")
    ax.set_title("cDSP-BO convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Running cDSP-based Bayesian Optimisation example...")
    X, Y = run_optimisation(n_iterations=100, target=10.0, emi_target=10.0)

    print("\nFinal best y:", float(np.min(Y)))
    print("Total samples:", len(X))

    plot_samples(X, n_initial=10, save_path=os.path.join(fig_dir, "samples.png"))
    plot_convergence(Y, save_path=os.path.join(fig_dir, "convergence.png"))
    print("Done.")
