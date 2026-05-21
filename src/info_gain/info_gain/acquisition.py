r"""Mutual-information acquisition for GP informative path planning.

This module is the SINGLE source of truth for the planner scoring math.
Both the ROS2 planner scripts (info_gain + nonstationary_planning) and
the standalone sweep import from here.

Reference
---------
Krause, A., Singh, A., & Guestrin, C. (2008). *Near-Optimal Sensor
Placements in Gaussian Processes: Theory, Efficient Algorithms and
Empirical Studies.* Journal of Machine Learning Research, 9(2), 235-284.

Why we use Krause-MI instead of pointwise info-gain
--------------------------------------------------
The textbook IPP score `Δ(x) = ½ log(1 + σ²(x)/σ_n²)` (Krause & Guestrin
2007 §3) measures uncertainty AT candidate x only. Late in a coverage
trial σ²(x) drops everywhere → all info-gains shrink → corner clustering.

The 2008 paper's mutual-information score fixes this by measuring the
MARGINAL GAIN in total field MI from observing x. A candidate next to an
existing sample contributes near-zero (its information is already "spent"),
so the score self-penalises clustering without any threshold knobs.

Math — TRUE Krause (2008) formula
----------------------------------
Marginal MI gain from adding noisy observation at x to set A
(Krause, Singh, Guestrin 2008, §3.2, Algorithm 1, Theorem 7):

    δ_MI(x | A) = ½ Σ_{y ∈ V} log[ (σ_n² + σ²(y|A)) / (σ_n² + σ²(y|A∪{x})) ]

By Sherman-Morrison, the posterior variance update at y after observing x
with noise σ_n² is:

    σ²(y | A∪{x}) = σ²(y|A) − ρ²(y, x|A) / (σ²(x|A) + σ_n²)

where ρ(y, x | A) = k(y, x) − K_yA (K_AA + σ_n² I)⁻¹ K_Ax is the
predictive cross-covariance, and σ²(x|A) is the LATENT posterior variance
(GPyTorch pred.variance). The denominator must include σ_n² — this is the
full predictive variance at the candidate location.

The log form is what Theorem 7 proves the (1−1/e) near-optimality bound
for. It is always ≥ 0 (observing x can only reduce field variance). Same
O(|V|·|A|) cost as any alternative.

We use V = the candidate grid (Krause's discrete-domain setting, §3.2).

Properties (Theorem 7 of the paper):
- Submodular → greedy gives (1 − 1/e) ≈ 0.632 of the optimal MI.
- Always ≥ 0 (monotone non-decreasing in observation set).
- Naturally prevents clustering — see test_krause_mi.py.

For pose-aware variants
-----------------------
- pose_aware (MC): average MI over MC perturbations of candidate
  locations under N(0, σ_x² I). Same plumbing as the pre-fix MC.
- analytical_girard: Girard's expected-kernel formula extends to the
  cross-covariance blocks that enter ρ(y, x | A), so the same Sherman-
  Morrison structure applies — the only change is using E[k(x̃, ·)]
  in place of k(x, ·) wherever x is the perturbed candidate.

CPU/GPU execution
-----------------
All linear-algebra hot paths run on `gp.device` if the GP exposes one.
The public API is numpy-in / numpy-out, so callers (ROS planners, tests)
do not change.  Internally:

- Kernel matrices are built with torch on the GP's device.
- K_AA Cholesky uses `torch.linalg.cholesky_ex` (non-blocking error
  reporting — required because regular `cholesky` on CUDA raises async
  errors that surface far from the call site).
- All acquisition math runs in fp64 to match the numpy precision the
  existing tests expect (atol = 1e-6).  The matrices are small enough
  (≤ 1 GB even at chunk_size=50 000) that fp64 still runs in well under
  a second on a 4070 Ti.
- pose_aware MC chunking auto-scales: 50 000 on CUDA, 5 000 on CPU.

References
----------
- Krause, Singh, Guestrin (2008). JMLR 9: 235-284.
- Krause, Guestrin (2007). *Nonmyopic Active Learning of Gaussian
  Processes.* ICML 2007 §3 (the pointwise formula we replaced).
- Girard, Rasmussen, Quinonero-Candela, Murray-Smith (2003).
  *Gaussian Process Priors with Uncertain Inputs.* NeurIPS 2003.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch

from info_gain.expected_kernel import (                          # noqa: E402
    expected_K_rbf,
    expected_K_paciorek_global,
    expected_K_paciorek_spatial_gh,
    Sigma_from_global_paciorek,
    renyi_prefactor,
)


# ---------------------------------------------------------------------------
# Device + tensor helpers
# ---------------------------------------------------------------------------
def _get_device(gp) -> torch.device:
    """Return the GP's device, defaulting to CPU when not exposed."""
    dev = getattr(gp, "device", None)
    if isinstance(dev, torch.device):
        return dev
    if dev is None:
        return torch.device("cpu")
    return torch.device(dev)


# Default dtype for the acquisition layer.  We use fp64 for the small
# kernel matrices and the Cholesky solve, matching the previous numpy
# implementation's precision (existing tests use 1e-6 tolerance).  The
# matrices are at most (|V|, |chunk|) = (2401, 50000) so even fp64 is
# only ~1 GB and runs in well under a second on a 4070 Ti.
_ACQ_DTYPE = torch.float64


def _as_tensor(x, device: torch.device, dtype=_ACQ_DTYPE) -> torch.Tensor:
    """Convert numpy or torch input to a tensor on `device` with `dtype`."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _get_training_tensors(gp, device: torch.device
                           ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return GP training tensors on `device` upcast to `_ACQ_DTYPE`.

    The GP models store training data in fp32 (their internal compute
    dtype), but the acquisition layer runs in fp64.  We upcast here so
    every kernel-eval call sees consistent dtypes.
    """
    if hasattr(gp, "get_training_tensors"):
        X_t, y_t = gp.get_training_tensors()
        if X_t is None or len(X_t) == 0:
            return None, None
        return (X_t.to(device=device, dtype=_ACQ_DTYPE),
                y_t.to(device=device, dtype=_ACQ_DTYPE))
    # Fallback for stub GPs in tests (no torch tensors available).
    X_np, y_np = gp.get_training_data()
    if X_np is None or len(X_np) == 0:
        return None, None
    return (_as_tensor(X_np, device),
            _as_tensor(y_np, device))


def _gp_predict_var(gp, X: torch.Tensor) -> torch.Tensor:
    """Predict latent posterior variance at X, returned as torch tensor on X's device."""
    if hasattr(gp, "predict_torch"):
        _, var = gp.predict_torch(X)
    else:
        _, var = gp.predict(X)
    if not isinstance(var, torch.Tensor):
        var = torch.as_tensor(var, dtype=X.dtype, device=X.device)
    return var.to(device=X.device, dtype=X.dtype)


# ---------------------------------------------------------------------------
# Kernel evaluation (RBF or Gibbs, kernel-agnostic, torch-internal)
# ---------------------------------------------------------------------------
def _rbf_kernel_torch(X1: torch.Tensor, X2: torch.Tensor,
                       signal_var: float, ell: float) -> torch.Tensor:
    """RBF kernel matrix on torch: k(x, x') = σ_f² exp(-½ ||x-x'||²/ℓ²)."""
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)            # (N, M, D)
    sq = diff.pow(2).sum(dim=-1)                         # (N, M)
    return signal_var * torch.exp(-0.5 * sq / (ell ** 2))


def _gibbs_kernel_eval_torch(gibbs, X1: torch.Tensor,
                              X2: torch.Tensor) -> torch.Tensor:
    """Evaluate a Gibbs kernel on arbitrary point arrays.

    The Gibbs kernel module is trained as fp32, so we run its forward
    in fp32 and upcast the output to the caller's dtype (typically fp64
    for the acquisition layer).
    """
    target_dev = gibbs.basis_centers.device
    out_dtype = X1.dtype
    X1 = X1.to(device=target_dev, dtype=torch.float32)
    X2 = X2.to(device=target_dev, dtype=torch.float32)
    with torch.no_grad():
        K = gibbs.forward(X1, X2)
        if hasattr(K, "evaluate"):
            K = K.evaluate()
        elif hasattr(K, "to_dense"):
            K = K.to_dense()
    return K.to(dtype=out_dtype)


def _sdk_kernel_eval_torch(sdk_kernel, X1: torch.Tensor,
                              X2: torch.Tensor) -> torch.Tensor:
    """Evaluate the SpectralDiffusionKernel (heat-kernel on graph)
    on arbitrary continuous points.  SDK runs in fp64 internally; we
    cast back to caller dtype on return."""
    target_dev = sdk_kernel._device
    out_dtype = X1.dtype
    X1f = X1.to(device=target_dev, dtype=torch.float64)
    X2f = X2.to(device=target_dev, dtype=torch.float64)
    with torch.no_grad():
        K = sdk_kernel(X1f, X2f)
        if hasattr(K, "evaluate"):
            K = K.evaluate()
        elif hasattr(K, "to_dense"):
            K = K.to_dense()
    return K.to(device=X1.device, dtype=out_dtype)


def _kernel_eval_torch(gp, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """Torch-tensor kernel eval, dispatching to SDK / Gibbs / RBF as appropriate."""
    if hasattr(gp, "sdk_kernel"):
        K = _sdk_kernel_eval_torch(gp.sdk_kernel, X1, X2)
        return K.to(device=X1.device, dtype=X1.dtype)
    if hasattr(gp, "gibbs_kernel"):
        K = _gibbs_kernel_eval_torch(gp.gibbs_kernel, X1, X2)
        return K.to(device=X1.device, dtype=X1.dtype)
    ell = float(getattr(gp, "_learned_lengthscale",
                          getattr(gp, "lengthscale", 2.0)))
    signal_var = float(getattr(gp, "_learned_signal_var",
                                  getattr(gp, "signal_var", 1.0)))
    return _rbf_kernel_torch(X1, X2, signal_var, ell)


def _signal_var_of(gp) -> float:
    # SDK stores log_signal_var on its kernel; expose it transparently.
    if hasattr(gp, "sdk_kernel"):
        return float(torch.exp(gp.sdk_kernel._log_signal_var).item())
    return float(getattr(gp, "_learned_signal_var",
                          getattr(gp, "signal_var", 1.0)))


# ---------------------------------------------------------------------------
# Backwards-compatible numpy wrappers (tests + ad-hoc analysis import these)
# ---------------------------------------------------------------------------
def _rbf_kernel(X1: np.ndarray, X2: np.ndarray,
                signal_var: float, ell: float) -> np.ndarray:
    """Numpy-in / numpy-out RBF kernel (kept for tests)."""
    diff = X1[:, None, :] - X2[None, :, :]
    sq = (diff ** 2).sum(axis=-1)
    return signal_var * np.exp(-0.5 * sq / (ell ** 2))


def _gibbs_kernel_eval(gibbs, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Numpy-in / numpy-out Gibbs kernel (kept for tests)."""
    device = gibbs.basis_centers.device
    X1_t = torch.as_tensor(X1, dtype=torch.float32, device=device)
    X2_t = torch.as_tensor(X2, dtype=torch.float32, device=device)
    K = _gibbs_kernel_eval_torch(gibbs, X1_t, X2_t)
    return K.detach().cpu().numpy()


def _kernel_eval(gp, X1, X2) -> np.ndarray:
    """Numpy-in / numpy-out kernel eval (kept for tests)."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if hasattr(gp, "gibbs_kernel"):
        return _gibbs_kernel_eval(gp.gibbs_kernel, X1, X2)
    ell = float(getattr(gp, "_learned_lengthscale",
                          getattr(gp, "lengthscale", 2.0)))
    signal_var = float(getattr(gp, "_learned_signal_var",
                                  getattr(gp, "signal_var", 1.0)))
    return _rbf_kernel(X1, X2, signal_var, ell)


# ---------------------------------------------------------------------------
# Travel cost (unchanged)
# ---------------------------------------------------------------------------
def travel_cost(candidates: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(candidates) -
                           np.asarray(current)[None, :], axis=1)


def score_with_travel(score: np.ndarray, candidates: np.ndarray,
                       current: np.ndarray, lambda_cost: float) -> np.ndarray:
    """score(x) = base_score(x) − λ · ‖x − current‖."""
    return np.asarray(score) - lambda_cost * travel_cost(candidates, current)


# ---------------------------------------------------------------------------
# Pointwise info-gain (kept ONLY for diagnostic logging — not for scoring)
# ---------------------------------------------------------------------------
def info_gain_pointwise(variance, noise_var: float):
    """Δ(x) = ½ log(1 + σ²(x)/σ_n²).

    This is the OLD acquisition function. Kept here so planners can
    log a `realized_info` column comparable across versions, NOT for
    making decisions.
    """
    if isinstance(variance, torch.Tensor):
        return 0.5 * torch.log1p(variance / noise_var)
    return 0.5 * np.log1p(np.asarray(variance) / noise_var)


# ---------------------------------------------------------------------------
# Krause-MI core (torch-internal)
# ---------------------------------------------------------------------------
def _solve_with_K_AA(K_AA: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve K_AA · X = B robustly.

    Uses `cholesky_ex` (non-blocking error reporting — important on CUDA
    where regular cholesky raises async errors).  Falls back to a generic
    linear solve on factorisation failure.  K_AA is upcast to fp64 first
    for stability on small near-PD matrices; B is solved at the same
    dtype and the result is cast back to B's original dtype.
    """
    in_dtype = B.dtype
    K_AA_64 = K_AA.to(torch.float64)
    B_64 = B.to(torch.float64)

    L, info = torch.linalg.cholesky_ex(K_AA_64)
    if torch.any(info != 0):
        # Fall back to a general linear solve.
        sol = torch.linalg.solve(K_AA_64, B_64)
    else:
        sol = torch.cholesky_solve(B_64, L)
    return sol.to(dtype=in_dtype)


def _predictive_covariance_torch(gp, Y: torch.Tensor, X: torch.Tensor,
                                  noise_var: float,
                                  X_train: Optional[torch.Tensor]
                                  ) -> torch.Tensor:
    """ρ(y, x | A) = k(y, x) − K_yA (K_AA + σ_n² I)⁻¹ K_Ax (torch-internal).

    Returns a torch tensor of shape (|Y|, |X|) on Y.device.

    Args:
        gp:        GP exposing kernel hyperparameters (or a Gibbs kernel).
        Y, X:      Float32 tensors on the same device.
        noise_var: σ_n² added to K_AA.
        X_train:   Pre-fetched training tensor or None for prior-only.
    """
    K_yx = _kernel_eval_torch(gp, Y, X)
    if X_train is None or len(X_train) == 0:
        return K_yx

    K_yA = _kernel_eval_torch(gp, Y, X_train)
    K_Ax = _kernel_eval_torch(gp, X_train, X)
    K_AA = _kernel_eval_torch(gp, X_train, X_train)
    n_a = X_train.shape[0]
    eye = torch.eye(n_a, dtype=K_AA.dtype, device=K_AA.device)
    # Slightly looser jitter (1e-5 vs numpy's 1e-6) for fp32 PD safety.
    jitter = 1e-5 if K_AA.dtype == torch.float32 else 1e-6
    K_AA = K_AA + (noise_var + jitter) * eye

    K_AA_inv_K_Ax = _solve_with_K_AA(K_AA, K_Ax)
    return K_yx - K_yA @ K_AA_inv_K_Ax


def predictive_covariance(gp, Y: np.ndarray, X: np.ndarray,
                           noise_var: Optional[float] = None) -> np.ndarray:
    """Numpy-in / numpy-out wrapper around the torch implementation.

    ρ(y, x | A) = k(y, x) − K_yA (K_AA + σ_n² I)⁻¹ K_Ax.
    Returns shape (|Y|, |X|).  Falls back to k(Y, X) when |A| = 0.
    """
    device = _get_device(gp)
    if noise_var is None:
        noise_var = float(getattr(gp, "noise_var", 0.36))
    Y_t = _as_tensor(Y, device)
    X_t = _as_tensor(X, device)
    X_tr, _ = _get_training_tensors(gp, device)
    rho = _predictive_covariance_torch(gp, Y_t, X_t,
                                        float(noise_var), X_tr)
    return rho.detach().cpu().numpy()


def _mi_score_torch(gp, candidates: torch.Tensor,
                     noise_var: float,
                     V: torch.Tensor) -> torch.Tensor:
    """Torch-internal Krause MI score (returns (N_cand,) tensor on candidates.device)."""
    device = candidates.device
    # σ²(x|A): predictive variance at candidates.
    sigma_sq_x = _gp_predict_var(gp, candidates)
    sigma_sq_x = sigma_sq_x.clamp(min=1e-9)

    # σ²(y|A): predictive variance at V-points (reuse if V is candidates).
    if V is candidates:
        sigma_sq_V = sigma_sq_x
    else:
        sigma_sq_V = _gp_predict_var(gp, V).clamp(min=1e-9)

    X_train, _ = _get_training_tensors(gp, device)

    # ρ(y, x | A): (|V|, N_cand)
    rho = _predictive_covariance_torch(gp, V, candidates,
                                        noise_var, X_train)

    # Sherman-Morrison: σ²(y|A∪{x}) = σ²(y|A) - ρ²(y,x|A)/(σ²(x|A)+σ_n²)
    denom = (sigma_sq_x + noise_var).unsqueeze(0)        # (1, N_cand)
    sigma_sq_after = (sigma_sq_V.unsqueeze(1)
                       - (rho ** 2) / denom).clamp(min=1e-9)

    # Krause (2008) §3.2: ½ Σ_y log[(σ_n²+σ²(y|A)) / (σ_n²+σ²(y|A∪{x}))]
    log_ratio = 0.5 * torch.log(
        (noise_var + sigma_sq_V.unsqueeze(1))
        / (noise_var + sigma_sq_after))
    return log_ratio.sum(dim=0)                          # (N_cand,)


def mi_score(gp, candidates: np.ndarray,
             noise_var: float,
             field_points: Optional[np.ndarray] = None) -> np.ndarray:
    """True Krause-Singh-Guestrin (2008) marginal MI score (numpy in / numpy out).

    For each candidate x, returns the marginal MI gain from adding a noisy
    observation at x to the current observation set A:

        δ_MI(x|A) = ½ Σ_{y∈V} log[(σ_n²+σ²(y|A)) / (σ_n²+σ²(y|A∪{x}))]

    Posterior-variance update (Sherman-Morrison):

        σ²(y|A∪{x}) = σ²(y|A) − ρ²(y,x|A) / (σ²(x|A) + σ_n²)

    The math runs on `gp.device` if the GP exposes one (torch tensors
    internally); otherwise it stays on CPU.  The return is always numpy.

    Args:
        gp: GP model.  See module docstring.
        candidates: (N, D) candidate locations.
        noise_var:  observation-noise variance σ_n².
        field_points: V in the formula.  Defaults to `candidates`.

    Returns:
        scores: (N,) numpy array, all >= 0.
    """
    device = _get_device(gp)
    cand_t = _as_tensor(candidates, device)
    if field_points is None:
        V_t = cand_t
    else:
        V_t = _as_tensor(field_points, device)
    scores = _mi_score_torch(gp, cand_t, float(noise_var), V_t)
    return scores.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Pose-aware Monte-Carlo wrapper
# ---------------------------------------------------------------------------
def _chunk_size_for_device(device: torch.device) -> int:
    """50 000 on CUDA (200 MB peak, fits 12 GB), 5 000 on CPU (100 MB)."""
    return 50000 if device.type == "cuda" else 5000


def mi_score_pose_aware(gp, candidates: np.ndarray,
                         noise_var: float, sigma_x: float,
                         n_mc: int = 30,
                         rng: Optional[np.random.Generator] = None,
                         field_points: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """MC expected MI score under isotropic position uncertainty.

    For each candidate x, draw n_mc perturbations ξ ~ N(0, σ_x² I), evaluate
    mi_score at x + ξ, and average.

    Internally evaluates on `gp.device` (CUDA if available).  Chunks the
    noisy candidates so the (|V|, |chunk|) covariance matrix stays in
    memory — chunk size scales with device.
    """
    candidates = np.asarray(candidates, dtype=float)
    N = len(candidates)
    if rng is None:
        rng = np.random.default_rng()
    if sigma_x <= 0.0 or n_mc <= 1:
        return mi_score(gp, candidates, noise_var, field_points)

    # Generate noisy candidates in numpy (RNG stream stays reproducible
    # across CPU/GPU paths).
    noise = rng.normal(0.0, sigma_x, size=(N * n_mc, 2))
    repeated = np.repeat(candidates, n_mc, axis=0)
    noisy = repeated + noise

    device = _get_device(gp)
    if field_points is None:
        V_t = _as_tensor(candidates, device)
    else:
        V_t = _as_tensor(field_points, device)

    chunk_size = _chunk_size_for_device(device)
    total = N * n_mc
    scores_flat = np.empty(total, dtype=float)
    for i in range(0, total, chunk_size):
        chunk_np = noisy[i:i + chunk_size]
        chunk_t = _as_tensor(chunk_np, device)
        scores_t = _mi_score_torch(gp, chunk_t, float(noise_var), V_t)
        scores_flat[i:i + chunk_size] = scores_t.detach().cpu().numpy()
    return scores_flat.reshape(N, n_mc).mean(axis=1)


# ---------------------------------------------------------------------------
# Kernel-side input-noise Krause-MI (Girard / Paciorek / Spatial Paciorek)
# ---------------------------------------------------------------------------
# All three planners share the same Sherman-Morrison + Krause-log structure;
# they differ only in how E[k(x̃, x_j)] is computed for the candidate-side
# cross blocks (K_yx, K_Ax).  K_AA and K_yA stay deterministic per Girard
# 2003 Eq. 12 (first-moment approximation).
#
# `utility` selects:
#   "shannon"  → ½ log(...) coefficient (default; matches Krause 2008).
#   "renyi"    → (σ_x² + 0.5) coefficient with α(Σ_x) = 1 + 1/Tr(Σ_x)
#                from Popovic et al. 2019 Eq. 8.  Reduces to Shannon as
#                σ_x → 0 (renyi_prefactor in expected_kernel.py).
def _mi_log_prefactor(sigma_x: float, utility: str) -> float:
    if utility == "shannon":
        return 0.5
    if utility == "renyi":
        return renyi_prefactor(sigma_x)
    raise ValueError(f"unknown utility '{utility}', expected 'shannon' or 'renyi'")


def _krause_mi_with_expected_blocks_torch(
    *,
    gp,
    V: torch.Tensor,
    K_yx_exp: torch.Tensor,
    K_Ax_exp: torch.Tensor,
    K_yA: torch.Tensor,
    K_AA_with_noise: torch.Tensor,
    signal_var: float,
    noise_var: float,
    sigma_x: float,
    utility: str,
) -> torch.Tensor:
    """Krause-MI core with pre-computed expected-kernel blocks.

    Inputs:
        V                 (|V|, 2) field points (deterministic).
        K_yx_exp          (|V|, N_cand) expected cross-covariance.
        K_Ax_exp          (|A|, N_cand) expected cross-covariance.
        K_yA              (|V|, |A|) deterministic kernel matrix.
        K_AA_with_noise   (|A|, |A|) deterministic K_AA + σ_n² I + jitter.
        signal_var, noise_var, sigma_x: scalars.
        utility           "shannon" or "renyi".
    Returns:
        (N_cand,) MI scores.
    """
    K_AA_inv_K_Ax = _solve_with_K_AA(K_AA_with_noise, K_Ax_exp)
    rho = K_yx_exp - K_yA @ K_AA_inv_K_Ax                      # (|V|, N_cand)

    # σ²(x̃|A): Girard first-moment approximation (Eq. 12).
    sigma_sq_exp = signal_var - (K_Ax_exp * K_AA_inv_K_Ax).sum(dim=0)
    sigma_sq_exp = sigma_sq_exp.clamp(min=1e-9)

    # σ²(y|A): EXACT predictive variance at V-points.
    sigma_sq_V = _gp_predict_var(gp, V).clamp(min=1e-9)

    # Sherman-Morrison + Krause log.
    denom = (sigma_sq_exp + noise_var).unsqueeze(0)
    sigma_sq_after = (sigma_sq_V.unsqueeze(1)
                       - rho ** 2 / denom).clamp(min=1e-9)
    pref = _mi_log_prefactor(sigma_x, utility)
    log_ratio = pref * torch.log(
        (noise_var + sigma_sq_V.unsqueeze(1))
        / (noise_var + sigma_sq_after))
    return log_ratio.sum(dim=0)


def _mi_score_girard_torch(gp, candidates: torch.Tensor,
                            noise_var: float, sigma_x: float,
                            V: torch.Tensor,
                            utility: str = "shannon") -> torch.Tensor:
    """Closed-form Girard-expected MI on torch (RBF kernel)."""
    device = candidates.device
    X_train, _ = _get_training_tensors(gp, device)
    if X_train is None or len(X_train) == 0:
        rng = np.random.default_rng()
        cand_np = candidates.detach().cpu().numpy()
        scores_np = mi_score_pose_aware(gp, cand_np, noise_var, sigma_x,
                                          n_mc=20, rng=rng)
        return torch.as_tensor(scores_np, dtype=candidates.dtype, device=device)

    ell = float(getattr(gp, "_learned_lengthscale",
                          getattr(gp, "lengthscale", 2.0)))
    signal_var = float(getattr(gp, "_learned_signal_var",
                                  getattr(gp, "signal_var", 1.0)))

    # E[k(y, x̃)]  shape (|V|, N_cand) — built via expected_K_rbf which
    # returns (N_cand, |V|), so transpose.
    K_yx_exp = expected_K_rbf(candidates, V, ell, signal_var, sigma_x).T
    # E[k(X_train, x̃)]  shape (|A|, N_cand)
    K_Ax_exp = expected_K_rbf(candidates, X_train, ell, signal_var, sigma_x).T

    # Deterministic K_yA, K_AA.
    K_yA = _rbf_kernel_torch(V, X_train, signal_var, ell)
    K_AA = _rbf_kernel_torch(X_train, X_train, signal_var, ell)
    n_a = X_train.shape[0]
    eye = torch.eye(n_a, dtype=K_AA.dtype, device=K_AA.device)
    jitter = 1e-5 if K_AA.dtype == torch.float32 else 1e-6
    K_AA = K_AA + (noise_var + jitter) * eye

    return _krause_mi_with_expected_blocks_torch(
        gp=gp, V=V, K_yx_exp=K_yx_exp, K_Ax_exp=K_Ax_exp,
        K_yA=K_yA, K_AA_with_noise=K_AA,
        signal_var=signal_var, noise_var=noise_var,
        sigma_x=sigma_x, utility=utility)


def _mi_score_paciorek_torch(gp, candidates: torch.Tensor,
                               noise_var: float, sigma_x: float,
                               V: torch.Tensor,
                               utility: str = "shannon") -> torch.Tensor:
    """Closed-form Girard-expected MI for the GLOBAL anisotropic Paciorek.

    Uses expected_K_paciorek_global() to substitute E[k(x̃, ·)] into
    K_yx and K_Ax.  K_AA and K_yA are evaluated by the existing
    Paciorek kernel module (deterministic).
    """
    device = candidates.device
    X_train, _ = _get_training_tensors(gp, device)
    if X_train is None or len(X_train) == 0:
        rng = np.random.default_rng()
        cand_np = candidates.detach().cpu().numpy()
        scores_np = mi_score_pose_aware(gp, cand_np, noise_var, sigma_x,
                                          n_mc=20, rng=rng)
        return torch.as_tensor(scores_np, dtype=candidates.dtype, device=device)

    # The PaciorekGPModel exposes its kernel module.  Σ_kernel is in
    # precision form (M); convert to covariance once.
    kernel = gp.kernel  # GlobalPaciorekKernel
    Sigma_kernel = Sigma_from_global_paciorek(kernel).to(
        device=device, dtype=candidates.dtype)
    signal_var = float(torch.exp(kernel._log_signal_var).detach().item())

    # Expected blocks
    K_yx_exp = expected_K_paciorek_global(
        candidates, V, Sigma_kernel, signal_var, sigma_x).T
    K_Ax_exp = expected_K_paciorek_global(
        candidates, X_train, Sigma_kernel, signal_var, sigma_x).T

    # Deterministic blocks via the kernel module (it already implements
    # the anisotropic Paciorek formula).
    with torch.no_grad():
        K_yA = kernel.forward(V.to(dtype=candidates.dtype),
                                X_train.to(dtype=candidates.dtype))
        K_AA = kernel.forward(X_train.to(dtype=candidates.dtype),
                                X_train.to(dtype=candidates.dtype))
    K_yA = K_yA.to(device=device, dtype=candidates.dtype)
    K_AA = K_AA.to(device=device, dtype=candidates.dtype)
    n_a = X_train.shape[0]
    eye = torch.eye(n_a, dtype=K_AA.dtype, device=K_AA.device)
    jitter = 1e-5 if K_AA.dtype == torch.float32 else 1e-6
    K_AA = K_AA + (noise_var + jitter) * eye

    return _krause_mi_with_expected_blocks_torch(
        gp=gp, V=V, K_yx_exp=K_yx_exp, K_Ax_exp=K_Ax_exp,
        K_yA=K_yA, K_AA_with_noise=K_AA,
        signal_var=signal_var, noise_var=noise_var,
        sigma_x=sigma_x, utility=utility)


def _mi_score_paciorek_spatial_torch(gp, candidates: torch.Tensor,
                                       noise_var: float, sigma_x: float,
                                       V: torch.Tensor,
                                       utility: str = "shannon",
                                       domain_min: float = 0.0,
                                       domain_max: float = 25.0
                                       ) -> torch.Tensor:
    """5-pt Gauss-Hermite expected MI for the SPATIAL Paciorek kernel.

    Σ(x) varies across the domain → no closed form; integrate the
    deterministic kernel via 25-node 2D quadrature per (candidate,
    target) pair.  Cost: 25× the deterministic kernel cost at the
    candidate set.
    """
    device = candidates.device
    X_train, _ = _get_training_tensors(gp, device)
    if X_train is None or len(X_train) == 0:
        rng = np.random.default_rng()
        cand_np = candidates.detach().cpu().numpy()
        scores_np = mi_score_pose_aware(gp, cand_np, noise_var, sigma_x,
                                          n_mc=20, rng=rng)
        return torch.as_tensor(scores_np, dtype=candidates.dtype, device=device)

    kernel = gp.kernel  # SpatialPaciorekKernel

    # Define a forward closure so the GH integrator can call it.
    def _forward(X1, X2):
        with torch.no_grad():
            K = kernel.forward(X1.to(dtype=candidates.dtype),
                                X2.to(dtype=candidates.dtype))
        return K.to(device=device, dtype=candidates.dtype)

    # Domain bounds: pull from kernel if available, else assume sweep default.
    dm = float(getattr(kernel, "domain_min", domain_min))
    dx = float(getattr(kernel, "domain_max", domain_max))

    # Chunk over candidates: GH quadrature builds an internal tensor of
    # shape (N_chunk * 25, |V|), which is the memory bottleneck.
    # N_cand=2401, V=2401, fp64 → ~1.15 GiB if unchunked.  500 cands per
    # chunk → ~240 MB, fits on a 16 GB GPU comfortably.
    n_cand = candidates.shape[0]
    chunk = 500 if device.type == "cuda" else 200
    K_yx_exp_chunks = []
    K_Ax_exp_chunks = []
    for i in range(0, n_cand, chunk):
        cand_i = candidates[i:i + chunk]
        K_yx_exp_chunks.append(expected_K_paciorek_spatial_gh(
            cand_i, V, _forward, sigma_x,
            domain_min=dm, domain_max=dx).T)
        K_Ax_exp_chunks.append(expected_K_paciorek_spatial_gh(
            cand_i, X_train, _forward, sigma_x,
            domain_min=dm, domain_max=dx).T)
    K_yx_exp = torch.cat(K_yx_exp_chunks, dim=1)
    K_Ax_exp = torch.cat(K_Ax_exp_chunks, dim=1)

    K_yA = _forward(V, X_train)
    K_AA = _forward(X_train, X_train)
    n_a = X_train.shape[0]
    eye = torch.eye(n_a, dtype=K_AA.dtype, device=K_AA.device)
    jitter = 1e-5 if K_AA.dtype == torch.float32 else 1e-6
    K_AA = K_AA + (noise_var + jitter) * eye

    signal_var = float(torch.exp(kernel._log_signal_var).detach().item())
    return _krause_mi_with_expected_blocks_torch(
        gp=gp, V=V, K_yx_exp=K_yx_exp, K_Ax_exp=K_Ax_exp,
        K_yA=K_yA, K_AA_with_noise=K_AA,
        signal_var=signal_var, noise_var=noise_var,
        sigma_x=sigma_x, utility=utility)


# ---------------------------------------------------------------------------
# Public numpy-in / numpy-out wrappers
# ---------------------------------------------------------------------------
def mi_score_girard(gp, candidates: np.ndarray,
                     noise_var: float, sigma_x: float,
                     field_points: Optional[np.ndarray] = None,
                     utility: str = "shannon",
                     ) -> np.ndarray:
    """True Krause (2008) MI with Girard (2003) expected RBF kernel.

    Falls back to `mi_score` when σ_x = 0.
    """
    candidates = np.asarray(candidates, dtype=float)
    if sigma_x <= 0.0:
        return mi_score(gp, candidates, noise_var, field_points)
    device = _get_device(gp)
    cand_t = _as_tensor(candidates, device)
    if field_points is None:
        V_t = cand_t
    else:
        V_t = _as_tensor(field_points, device)
    scores = _mi_score_girard_torch(gp, cand_t, float(noise_var),
                                      float(sigma_x), V_t,
                                      utility=utility)
    return scores.detach().cpu().numpy()


def mi_score_paciorek(gp, candidates: np.ndarray,
                       noise_var: float, sigma_x: float,
                       field_points: Optional[np.ndarray] = None,
                       utility: str = "shannon",
                       ) -> np.ndarray:
    """Krause (2008) MI with Girard expected GLOBAL anisotropic Paciorek kernel.

    Closed form.  Reduces to RBF Girard when the kernel anisotropy
    Σ_kernel = ℓ² I.  Falls back to `mi_score` when σ_x = 0.
    """
    candidates = np.asarray(candidates, dtype=float)
    if sigma_x <= 0.0:
        return mi_score(gp, candidates, noise_var, field_points)
    device = _get_device(gp)
    cand_t = _as_tensor(candidates, device)
    V_t = cand_t if field_points is None else _as_tensor(field_points, device)
    scores = _mi_score_paciorek_torch(gp, cand_t, float(noise_var),
                                        float(sigma_x), V_t,
                                        utility=utility)
    return scores.detach().cpu().numpy()


def mi_score_paciorek_spatial(gp, candidates: np.ndarray,
                                noise_var: float, sigma_x: float,
                                field_points: Optional[np.ndarray] = None,
                                utility: str = "shannon",
                                ) -> np.ndarray:
    """Krause (2008) MI with 5-pt GH expected SPATIAL Paciorek kernel.

    Cost: 25× deterministic kernel evals at candidates.  Falls back to
    `mi_score` when σ_x = 0.
    """
    candidates = np.asarray(candidates, dtype=float)
    if sigma_x <= 0.0:
        return mi_score(gp, candidates, noise_var, field_points)
    device = _get_device(gp)
    cand_t = _as_tensor(candidates, device)
    V_t = cand_t if field_points is None else _as_tensor(field_points, device)
    scores = _mi_score_paciorek_spatial_torch(gp, cand_t, float(noise_var),
                                                float(sigma_x), V_t,
                                                utility=utility)
    return scores.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Self-tests (smoke; full tests live in tests/test_krause_mi.py)
# ---------------------------------------------------------------------------
def _smoke() -> None:
    """Quick sanity test that the module imports and runs on CPU."""

    class FakeGP:
        """Minimal GP stub for self-test (RBF, learned params, CPU only)."""
        def __init__(self, X_train, y_train, noise_var=0.36, ell=2.0,
                     signal_var=1.0):
            self._X_train = np.asarray(X_train, dtype=float)
            self._y_train = np.asarray(y_train, dtype=float)
            self.noise_var = noise_var
            self._learned_lengthscale = ell
            self._learned_signal_var = signal_var
            self.device = torch.device("cpu")

        def get_training_data(self):
            return self._X_train, self._y_train

        def get_training_tensors(self):
            return (torch.as_tensor(self._X_train, dtype=torch.float32),
                    torch.as_tensor(self._y_train, dtype=torch.float32))

        def predict(self, X_test):
            X = np.asarray(X_test, dtype=float) if not isinstance(
                X_test, torch.Tensor) else X_test.detach().cpu().numpy()
            X_train = self._X_train
            ell = self._learned_lengthscale
            sv = self._learned_signal_var
            K_x_X = sv * np.exp(-0.5 *
                                  ((X[:, None, :] - X_train[None, :, :]) ** 2)
                                  .sum(-1) / ell ** 2)
            K_XX = sv * np.exp(-0.5 *
                                 ((X_train[:, None, :] - X_train[None, :, :])
                                  ** 2).sum(-1) / ell ** 2)
            K_XX = K_XX + (self.noise_var + 1e-6) * np.eye(len(X_train))
            mean = K_x_X @ np.linalg.solve(K_XX, self._y_train)
            var = sv - np.einsum(
                'ij,ji->i', K_x_X,
                np.linalg.solve(K_XX, K_x_X.T))
            return mean, np.maximum(var, 1e-9)

        def predict_torch(self, X_test):
            mean, var = self.predict(X_test)
            return (torch.as_tensor(mean, dtype=torch.float32),
                    torch.as_tensor(var, dtype=torch.float32))

    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 10, size=(15, 2))
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
    gp = FakeGP(X_train, y_train)

    cands = rng.uniform(0, 10, size=(50, 2))
    s = mi_score(gp, cands, noise_var=0.36)
    assert s.shape == (50,)
    assert np.all(np.isfinite(s))
    assert np.all(s >= -1e-6)
    print(f"[smoke] mi_score range = [{s.min():.4f}, {s.max():.4f}]")

    # Girard at sigma_x=0 should match mi_score (within tight tol).
    s_g = mi_score_girard(gp, cands, noise_var=0.36, sigma_x=0.0)
    rel = np.abs(s_g - s).max() / max(s.max(), 1e-9)
    print(f"[smoke] girard(σ=0) vs mi_score: max relative diff = {rel:.6f}")
    assert rel < 1e-5

    # Pose-aware MC at σ=0 should match exactly.
    s_pa = mi_score_pose_aware(gp, cands, noise_var=0.36, sigma_x=0.0,
                                 rng=np.random.default_rng(1))
    assert np.allclose(s_pa, s, atol=1e-6)
    print("[smoke] mi_score_pose_aware(σ=0) ≡ mi_score: ok")

    print("[smoke] OK")


if __name__ == '__main__':
    _smoke()
