r"""Expected-kernel functions under Gaussian input noise.

This module is the SINGLE source of truth for "what is E[k(x̃, x_j)]
when x̃ ~ N(μ, Σ_x) and the kernel is..."

Three families implemented:

1. expected_K_rbf
   - Closed form (Girard et al. 2003, Eq. 8):
     E[k(x̃, x_j)] = σ_f²/(1 + σ_x²/ℓ²) · exp(-½ ‖μ-x_j‖² / (ℓ² + σ_x²))
   - Special case of (2) when Σ_kernel = ℓ² I.

2. expected_K_paciorek_global
   - Closed form for the GLOBAL anisotropic Paciorek (constant Σ_kernel):
     E[k(x̃, x_j)] = σ_f² · √(det Σ / det A) · exp(-½ d^T A⁻¹ d)
     where A = Σ_kernel + Σ_x and d = μ - x_j.
   - Reduces to expected_K_rbf when Σ_kernel = ℓ² I.

3. expected_K_paciorek_spatial_gh
   - 5-point Gauss-Hermite quadrature in 2D (= 25 evals/pair) for the
     SPATIAL Paciorek kernel where Σ(x) varies across the domain.
   - Same approach as Jadidi et al. (2018) and Popovic et al. (2019).
   - Falls back to integrating the existing deterministic kernel.

Convention
----------
All functions take **Σ_kernel (covariance form)** as input — even though
GlobalPaciorekKernel internally stores M = Σ_kernel⁻¹ (precision form).
At planner entry the orchestrator inverts M once explicitly.  This is
intentional — the closed-form formula is cleanest in Σ-space, and the
Σ ↔ M flip is the single most likely source of algebra bugs.  Test 2
(σ_x → 0 reduction) catches direction flips.

For the spatial kernel, Σ(x) is read directly from the kernel module
(it stores Σ at anchors, not M).

Block substitution
------------------
We mirror Girard 2003 Eq. 12 exactly: only the candidate-side cross
blocks (K_yx, K_Ax) get the expected-kernel substitution.  K_yA and
K_AA stay deterministic.  The predictive variance at the noisy
candidate is then σ²(x̃|A) ≈ σ_f² − E[K_Ax̃]^T (K_AA + σ_n² I)⁻¹ E[K_Ax̃]
(Girard's first-moment approximation).

References
----------
- Girard, A., Rasmussen, C. E., Quinonero-Candela, J., & Murray-Smith,
  R. (2003). Gaussian Process Priors with Uncertain Inputs. NeurIPS 2003.
  Eq. 8 (closed-form RBF expectation), Eq. 12 (first-moment variance).
- Paciorek, C. J., & Schervish, M. J. (2004). Nonstationary Covariance
  Functions for Gaussian Process Regression. NeurIPS 2003 / 2004.
- Jadidi, M. G., Valls Miró, J., & Dissanayake, G. (2018). Gaussian
  process autonomous mapping and exploration for range-sensing mobile
  robots. Autonomous Robots 42(2): 273-290.
- Popovic, M., Vidal-Calleja, T., Hitz, G., Chung, J. J., Sa, I.,
  Siegwart, R., & Nieto, J. (2019). An informative path planning
  framework for active learning in UAV-based semantic mapping under
  localization uncertainty. ICRA 2020 / arXiv 1902.09660. Eq. 4-9.

Tests
-----
Run `python3 -m info_gain.expected_kernel` to execute the 5 internal
sanity checks.  All must pass before any new planner is called.
"""
from __future__ import annotations
from typing import Optional, Tuple

import math
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DTYPE = torch.float64

# 5-point Gauss-Hermite (physicists' convention, weight exp(-x²)).
# Source: Abramowitz & Stegun §25.4.46.
_GH5_KNOTS_NUMPY = np.array([
    -2.020182870456085632929,
    -0.958572464613818507112,
     0.000000000000000000000,
     0.958572464613818507112,
     2.020182870456085632929,
], dtype=np.float64)
_GH5_WEIGHTS_NUMPY = np.array([
    0.019953242059045913208,
    0.393619323152241159828,
    0.945308720482941881226,
    0.393619323152241159828,
    0.019953242059045913208,
], dtype=np.float64)


def gh5_knots_weights() -> Tuple[np.ndarray, np.ndarray]:
    """Return 5-point Gauss-Hermite (knots, weights) as numpy fp64."""
    return _GH5_KNOTS_NUMPY.copy(), _GH5_WEIGHTS_NUMPY.copy()


# ---------------------------------------------------------------------------
# (1) RBF — closed form
# ---------------------------------------------------------------------------
def expected_K_rbf(mu: torch.Tensor, X_j: torch.Tensor,
                    ell: float, signal_var: float,
                    sigma_x: float) -> torch.Tensor:
    """E[k_rbf(x̃, x_j)] for x̃ ~ N(μ, σ_x² I).

    Closed form: σ_f² / (1 + σ_x²/ℓ²) · exp(-½ ‖μ-x_j‖² / (ℓ² + σ_x²)).

    Args:
        mu:         (N_cand, 2) candidate means.
        X_j:        (N_j, 2) deterministic target points.
        ell:        scalar lengthscale.
        signal_var: σ_f².
        sigma_x:    scalar input-noise std.

    Returns:
        (N_cand, N_j) expected-kernel matrix.
    """
    s2 = sigma_x ** 2
    ell2 = float(ell) ** 2
    A_lambda = ell2 + s2
    amp = float(signal_var) / (1.0 + s2 / ell2)
    diff = mu.unsqueeze(1) - X_j.unsqueeze(0)              # (N_cand, N_j, 2)
    sq = diff.pow(2).sum(dim=-1)                            # (N_cand, N_j)
    return amp * torch.exp(-0.5 * sq / A_lambda)


# ---------------------------------------------------------------------------
# (2) Global Paciorek — closed form
# ---------------------------------------------------------------------------
def _det_2x2(M: torch.Tensor) -> torch.Tensor:
    """det of a 2x2 tensor (last two dims). Closed-form for speed."""
    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]


def _inv_2x2(M: torch.Tensor, det: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Inverse of a 2x2 SPD tensor (last two dims). Closed-form."""
    if det is None:
        det = _det_2x2(M).clamp(min=1e-12)
    inv = torch.empty_like(M)
    inv[..., 0, 0] =  M[..., 1, 1] / det
    inv[..., 0, 1] = -M[..., 0, 1] / det
    inv[..., 1, 0] = -M[..., 1, 0] / det
    inv[..., 1, 1] =  M[..., 0, 0] / det
    return inv


def expected_K_paciorek_global(mu: torch.Tensor, X_j: torch.Tensor,
                                 Sigma_kernel: torch.Tensor,
                                 signal_var: float,
                                 sigma_x: float) -> torch.Tensor:
    """E[k_paciorek_global(x̃, x_j)] for x̃ ~ N(μ, σ_x² I).

    For globally anisotropic Paciorek (constant Σ_kernel across the
    domain) the kernel reduces to anisotropic-RBF:
        k(x, x') = σ_f² exp(-½ d^T Σ_kernel⁻¹ d).

    The Gaussian integral in d gives the closed form:
        amp = σ_f² · √(det Σ_kernel / det A)
        E[k] = amp · exp(-½ d^T A⁻¹ d),    A = Σ_kernel + σ_x² I

    This subsumes expected_K_rbf when Σ_kernel = ℓ² I (Test 1).

    Args:
        mu:           (N_cand, 2) candidate means.
        X_j:          (N_j, 2) deterministic target points.
        Sigma_kernel: (2, 2) kernel covariance (NOT precision).
        signal_var:   σ_f².
        sigma_x:      scalar input-noise std.

    Returns:
        (N_cand, N_j) expected-kernel matrix.
    """
    device = mu.device
    dtype = mu.dtype
    Sigma_kernel = Sigma_kernel.to(device=device, dtype=dtype)
    s2 = float(sigma_x) ** 2
    A = Sigma_kernel + s2 * torch.eye(2, device=device, dtype=dtype)
    det_S = _det_2x2(Sigma_kernel).clamp(min=1e-12)
    det_A = _det_2x2(A).clamp(min=1e-12)
    amp = float(signal_var) * torch.sqrt(det_S / det_A)
    A_inv = _inv_2x2(A, det_A)                              # (2, 2)
    d = mu.unsqueeze(1) - X_j.unsqueeze(0)                  # (N_cand, N_j, 2)
    # quad = d^T A_inv d  computed as einsum
    quad = torch.einsum('cjp,pq,cjq->cj', d, A_inv, d)
    return amp * torch.exp(-0.5 * quad)


# ---------------------------------------------------------------------------
# (3) Spatial Paciorek — Gauss-Hermite quadrature
# ---------------------------------------------------------------------------
def _gh5_offsets_2d(sigma_x: float, device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (offsets, weights) for 2D Gauss-Hermite under N(0, σ_x² I).

    For x ~ N(0, σ_x² I), the change of variable y = x/(√2 σ_x) maps the
    Gaussian density to the GH weight exp(-y²)/√π.  So the integrand
    evaluated at the original x is integrated with knots √2 σ_x · ξ
    and the GH-weights need a factor 1/√π PER DIMENSION → 1/π in 2D.

    Returns:
        offsets: (25, 2) on `device` with `dtype`.
        weights: (25,)  on `device` with `dtype`.
    """
    xi = torch.tensor(_GH5_KNOTS_NUMPY, device=device, dtype=dtype)
    w = torch.tensor(_GH5_WEIGHTS_NUMPY, device=device, dtype=dtype)
    # Cartesian product
    a, b = torch.meshgrid(xi, xi, indexing='ij')
    offsets = math.sqrt(2.0) * float(sigma_x) * torch.stack(
        [a.reshape(-1), b.reshape(-1)], dim=-1)              # (25, 2)
    wa, wb = torch.meshgrid(w, w, indexing='ij')
    weights = (wa * wb).reshape(-1) / math.pi                 # (25,)
    return offsets, weights


def expected_K_paciorek_spatial_gh(
    mu: torch.Tensor,
    X_j: torch.Tensor,
    forward_kernel,
    sigma_x: float,
    domain_min: float = -float("inf"),
    domain_max: float = float("inf"),
) -> torch.Tensor:
    """E[k_paciorek_spatial(x̃, x_j)] via 5-pt Gauss-Hermite quadrature in 2D.

    For x̃ ~ N(μ, σ_x² I), build 25 quadrature nodes around each μ and
    integrate the existing deterministic spatial Paciorek kernel.  The
    integrand evaluation re-uses `forward_kernel(nodes, X_j)` so the
    Σ(x)-interpolation logic lives in one place.

    Args:
        mu:             (N_cand, 2) candidate means.
        X_j:            (N_j, 2) deterministic target points.
        forward_kernel: callable (X1, X2) -> (|X1|, |X2|) tensor — the
            deterministic kernel forward.  Typically a closure around
            `SpatialPaciorekKernel.forward` with no_grad.
        sigma_x:        scalar input-noise std.
        domain_min, domain_max: clamp limits for GH nodes.  Spatial
            extrapolation outside the anchor hull is bounded but
            biased; clamp to [domain_min - σ_x, domain_max + σ_x].
            Default: no clamp (use ±inf).

    Returns:
        (N_cand, N_j) expected-kernel matrix.
    """
    device = mu.device
    dtype = mu.dtype
    N_cand = mu.shape[0]
    N_j = X_j.shape[0]

    offsets, weights = _gh5_offsets_2d(sigma_x, device, dtype)  # (25,2), (25,)
    # nodes: (N_cand, 25, 2)
    nodes = mu.unsqueeze(1) + offsets.unsqueeze(0)
    # Domain clamp.  Don't NaN the gradient — clamp in-place is fine here
    # since forward kernel uses no_grad.
    if math.isfinite(domain_min) or math.isfinite(domain_max):
        lo = domain_min - sigma_x
        hi = domain_max + sigma_x
        nodes = nodes.clamp(min=lo, max=hi)
    nodes_flat = nodes.reshape(-1, 2)                            # (N_cand*25, 2)

    # Evaluate the deterministic kernel at all nodes against X_j.
    K_full = forward_kernel(nodes_flat, X_j)                     # (N_cand*25, N_j)
    K_full = K_full.to(device=device, dtype=dtype).view(N_cand, 25, N_j)
    # E[k] = sum_k W_k · K_full[:, k, :]
    K_exp = torch.einsum('k,ckj->cj', weights, K_full)
    return K_exp


# ---------------------------------------------------------------------------
# Rényi prefactor (Popovic Eq. 8)
# ---------------------------------------------------------------------------
def renyi_prefactor(sigma_x: float) -> float:
    """Closed-form Rényi-α prefactor for Σ_x = σ_x² I_2.

    Popovic Eq. 8: α(Σ) = 1 + 1/Tr(Σ_x).  With Σ_x = σ_x² I_2,
    Tr(Σ_x) = 2 σ_x², so α = 1 + 1/(2 σ_x²).

    The Rényi-α entropy of a Gaussian, after the irrelevant constants
    cancel in the entropy *difference*, contributes a per-V term
    proportional to (α / (2(α-1))) · log(σ²_total).

    Algebra:
        α / (2(α-1))
        = (1 + 1/(2σ_x²)) / (2 · 1/(2σ_x²))
        = (1 + 1/(2σ_x²)) · σ_x²
        = σ_x² + 0.5

    Returns this scalar.  As σ_x → 0, prefactor → 0.5 (Shannon).
    Closed form, no division by tiny numbers.
    """
    return float(sigma_x) ** 2 + 0.5


# ---------------------------------------------------------------------------
# Σ ↔ M helper (used by orchestrator at planner entry)
# ---------------------------------------------------------------------------
def Sigma_from_global_paciorek(kernel) -> torch.Tensor:
    """Read GlobalPaciorekKernel runtime state and return Σ_kernel (2, 2).

    GlobalPaciorekKernel stores M = R(θ) · diag(1/l₁², 1/l₂²) · R(θ)ᵀ
    (precision form).  We need Σ = M⁻¹ = R(θ) · diag(l₁², l₂²) · R(θ)ᵀ.

    Reads via the public helper `kernel.lengthscales()` which already
    applies softplus + 1e-3.
    """
    l1, l2, theta_deg = kernel.lengthscales()
    theta = float(theta_deg) * math.pi / 180.0
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=_DTYPE)
    L = torch.tensor([[l1 ** 2, 0.0], [0.0, l2 ** 2]], dtype=_DTYPE)
    return R @ L @ R.T


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _test_1_rbf_reduction() -> None:
    """expected_K_paciorek_global with Σ = ℓ² I matches expected_K_rbf."""
    rng = np.random.default_rng(0)
    mu = torch.tensor(rng.uniform(0, 25, size=(8, 2)), dtype=_DTYPE)
    Xj = torch.tensor(rng.uniform(0, 25, size=(6, 2)), dtype=_DTYPE)
    cases = [(1.0, 0.5), (2.0, 2.0), (5.0, 3.0), (1.5, 0.0)]
    max_diff = 0.0
    for ell, sigma_x in cases:
        sf2 = 0.7
        S = (ell ** 2) * torch.eye(2, dtype=_DTYPE)
        a = expected_K_rbf(mu, Xj, ell, sf2, sigma_x)
        b = expected_K_paciorek_global(mu, Xj, S, sf2, sigma_x)
        diff = float((a - b).abs().max())
        max_diff = max(max_diff, diff)
        assert diff < 1e-9, f"Test 1 failed at ell={ell}, σ={sigma_x}: diff={diff}"
    print(f"  [1] RBF reduction: max abs diff = {max_diff:.2e}  ✓")


def _test_2_sigma_zero_limit() -> None:
    """At σ_x → 0, expected_K_paciorek_global matches deterministic kernel."""
    rng = np.random.default_rng(1)
    mu = torch.tensor(rng.uniform(0, 25, size=(5, 2)), dtype=_DTYPE)
    Xj = torch.tensor(rng.uniform(0, 25, size=(7, 2)), dtype=_DTYPE)
    # Anisotropic Σ_kernel
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=_DTYPE)
    L = torch.tensor([[2.5 ** 2, 0.0], [0.0, 1.2 ** 2]], dtype=_DTYPE)
    Sigma = R @ L @ R.T
    sf2 = 0.9
    Sigma_inv = torch.linalg.inv(Sigma)
    d = mu.unsqueeze(1) - Xj.unsqueeze(0)                         # (5, 7, 2)
    quad = torch.einsum('cjp,pq,cjq->cj', d, Sigma_inv, d)
    K_det = sf2 * torch.exp(-0.5 * quad)
    K_exp = expected_K_paciorek_global(mu, Xj, Sigma, sf2, sigma_x=1e-6)
    diff = float((K_det - K_exp).abs().max())
    assert diff < 1e-6, f"Test 2 failed: diff={diff}"
    print(f"  [2] σ_x → 0 limit:        max abs diff = {diff:.2e}  ✓")


def _test_3_gh_vs_global() -> None:
    """GH-quadrature with constant Σ(x) matches closed-form global."""
    # Build a "spatial" forward that returns the same kernel as global.
    rng = np.random.default_rng(2)
    mu = torch.tensor(rng.uniform(2, 23, size=(4, 2)), dtype=_DTYPE)
    Xj = torch.tensor(rng.uniform(2, 23, size=(4, 2)), dtype=_DTYPE)
    theta = 0.4
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=_DTYPE)
    L = torch.tensor([[2.0 ** 2, 0.0], [0.0, 1.5 ** 2]], dtype=_DTYPE)
    Sigma = R @ L @ R.T
    Sigma_inv = torch.linalg.inv(Sigma)
    sf2 = 1.0

    def forward_anisotropic_rbf(X1, X2):
        d = X1.unsqueeze(1) - X2.unsqueeze(0)
        quad = torch.einsum('ijp,pq,ijq->ij', d, Sigma_inv.to(X1.dtype), d)
        return sf2 * torch.exp(-0.5 * quad)

    sigma_x = 0.7
    K_closed = expected_K_paciorek_global(mu, Xj, Sigma, sf2, sigma_x)
    K_gh = expected_K_paciorek_spatial_gh(mu, Xj, forward_anisotropic_rbf,
                                            sigma_x)
    diff = float((K_closed - K_gh).abs().max())
    assert diff < 1e-4, f"Test 3 failed: diff={diff}"
    print(f"  [3] GH ≡ closed form:     max abs diff = {diff:.2e}  ✓")


def _test_4_psd_check() -> None:
    """Predictive covariance built from expected blocks is PSD on a small V."""
    rng = np.random.default_rng(3)
    V = torch.tensor(rng.uniform(0, 25, size=(8, 2)), dtype=_DTYPE)
    A = torch.tensor(rng.uniform(0, 25, size=(5, 2)), dtype=_DTYPE)
    cand = torch.tensor(rng.uniform(0, 25, size=(1, 2)), dtype=_DTYPE)
    # Use isotropic RBF as the kernel for the smoke
    ell = 2.0
    sf2 = 1.0
    Sigma = (ell ** 2) * torch.eye(2, dtype=_DTYPE)
    sigma_x = 1.5

    def kfn(X1, X2):
        d = X1.unsqueeze(1) - X2.unsqueeze(0)
        sq = d.pow(2).sum(dim=-1)
        return sf2 * torch.exp(-0.5 * sq / (ell ** 2))

    K_AA = kfn(A, A) + 0.36 * torch.eye(A.shape[0], dtype=_DTYPE)
    K_yA = kfn(V, A)
    # Use expected blocks at the candidate
    K_yc = expected_K_paciorek_global(cand, V, Sigma, sf2, sigma_x).T  # (V, 1)
    K_Ac = expected_K_paciorek_global(cand, A, Sigma, sf2, sigma_x).T  # (A, 1)

    # Build predictive cov of (V, candidate) via Schur complement.
    # Cov(V, V|A) = K_VV - K_VA K_AA⁻¹ K_AV   (deterministic part)
    K_VV = kfn(V, V)
    K_AA_inv_K_AV = torch.linalg.solve(K_AA, K_yA.T)
    Cov_VV = K_VV - K_yA @ K_AA_inv_K_AV
    # Cross between V and cand: K_yc - K_yA K_AA⁻¹ K_Ac
    K_AA_inv_K_Ac = torch.linalg.solve(K_AA, K_Ac)
    Cov_Vc = K_yc - K_yA @ K_AA_inv_K_Ac
    # Var at candidate (Girard first-moment approx)
    var_cc = sf2 - (K_Ac * K_AA_inv_K_Ac).sum(dim=0)
    # Stack
    top = torch.cat([Cov_VV, Cov_Vc], dim=1)
    bot = torch.cat([Cov_Vc.T, var_cc.unsqueeze(0)], dim=1)
    P = torch.cat([top, bot], dim=0)
    P = 0.5 * (P + P.T)  # symmetrise numerically
    eigmin = float(torch.linalg.eigvalsh(P).min())
    assert eigmin > -1e-9, f"Test 4 failed: predictive cov not PSD, eigmin={eigmin}"
    print(f"  [4] PSD check:            min eigval   = {eigmin:.2e}  ✓")


def _test_5_renyi_shannon_limit() -> None:
    """Rényi prefactor → 0.5 (Shannon) as σ_x → 0."""
    pref = renyi_prefactor(1e-3)
    diff = abs(pref - 0.5)
    assert diff < 1e-5, f"Test 5 failed: prefactor={pref}, expected 0.5"
    # And at σ_x = 1.0, prefactor should be exactly 1.5
    pref1 = renyi_prefactor(1.0)
    assert abs(pref1 - 1.5) < 1e-12, f"Test 5b failed: prefactor at σ=1 is {pref1}"
    print(f"  [5] Rényi → Shannon:      |pref-0.5|   = {diff:.2e}  ✓")


def _smoke() -> None:
    print("expected_kernel.py self-tests:")
    _test_1_rbf_reduction()
    _test_2_sigma_zero_limit()
    _test_3_gh_vs_global()
    _test_4_psd_check()
    _test_5_renyi_shannon_limit()
    print("expected_kernel.py: 5/5 tests passed")


if __name__ == "__main__":
    _smoke()
