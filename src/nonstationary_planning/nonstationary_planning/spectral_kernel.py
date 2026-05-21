"""Spectral self-consistent kernel update (replacement for MAP-Gibbs).

Motivation (JD's framework, Field Framework v2 §3-5):
  A kernel k(x, x') has a dual spectral density h(lambda_l) on the observation
  graph Laplacian. The self-consistent fixed point is h*(l) = h_0(l) * exp(-1 - T_l[h*])
  where T_l is the mutual-information source at mode l computed from the data.

Pragmatic reduction for a 2D SE kernel with lengthscales (l1, l2) and angle theta:
  The kernel's spectral density in Fourier space (wavenumber k_x, k_y) is a
  Gaussian with widths 1/l1, 1/l2 rotated by theta. So the kernel's shape is
  fully determined by the empirical second-moment tensor of the data residuals
  as a function of spatial lag.

Method (this file):
  1. Subtract the posterior mean from observations to get residuals r_i.
  2. Compute all pairwise outer products r_i * r_j and spatial lags d_ij = x_j - x_i.
  3. Fit a positive-definite 2x2 "correlation-length tensor" L to an
     anisotropic SE model: corr(d) = exp(-0.5 d^T L^{-1} d).
  4. Return (l1, l2, theta) = eigen-decomposition of L.

Why this replaces MAP-Gibbs:
  MAP-Gibbs has 76 parameters and a log-normal prior pinned to l_init=2.0.
  With only ~100 samples and tau^2=1.5, the prior dominated and the kernel
  did not adapt to field anisotropy (l1/l2 -> 1 in every trial).

  The spectral fit has 3 parameters (l1, l2, theta), uses every pair of
  observations, and has no prior. On y_compress (narrow in y), we expect
  l_y < l_x to emerge directly from the data.

This is a GLOBAL anisotropy estimate — the returned (l1, l2, theta) is the
same for every spatial location. Localised kernel dynamics (a genuine
spatially-varying kernel) is a future extension; fixing the global anisotropy
is the first step and was the part that failed under MAP-Gibbs.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def fit_anisotropy(X: np.ndarray, y: np.ndarray,
                   signal_var: float = 1.0,
                   noise_var: float = 0.36,
                   max_pairs: int = 5000,
                   l_min: float = 0.5, l_max: float = 5.0,
                   n_angles: int = 8,
                   rng: "np.random.Generator | None" = None,
                   ) -> Tuple[float, float, float, dict]:
    """Fit (l1, l2, theta) via directional empirical variograms.

    Model (SE kernel):
        gamma(d) = 0.5 E[(Z(x+d) - Z(x))^2]  ~  sigma_f^2 (1 - exp(-|d|^2 / (2 l_theta^2)))
    where l_theta is the effective lengthscale in direction theta_d = atan2(dy, dx).

    For an anisotropic SE with principal axes (l1, l2) rotated by theta_p:
        1/l_theta^2 = cos^2(theta_d - theta_p)/l1^2 + sin^2(theta_d - theta_p)/l2^2

    Method:
      1. Bin pairs by direction into n_angles sectors covering [0, pi).
      2. Per sector, fit SE variogram via least squares at small-to-medium lags.
      3. Collect (l_theta for theta in sectors) -> fit l1, l2, theta_p by least
         squares on the ellipse 1/l_theta^2 = a cos^2(theta - theta_p) + b sin^2(...).

    Args:
        X, y: (N,2), (N,) training data
        noise_var: observation noise variance (known sensor spec)
        l_min, l_max: clip returned lengthscales
        n_angles: number of angular sectors
        rng: optional np.random.Generator for the pair-subsample step (B4
             plumbing — defaults to np.random.default_rng() if None).

    Returns:
        l1, l2, theta_rad, info dict.  When the fit is unreliable (sparse
        data, sill below noise floor, saturated lengthscales, ellipse
        residuals too large), returns (2.0, 2.0, 0.0, {reason: ...}) so
        the caller can SKIP the kernel update instead of writing garbage.

    Hardening (post-saturation-fix, 2026-04-29):
      The pre-fix version of this function would happily report
      l1≈l2≈l_max=5.0 on sparse data because:
        * Variogram sill collapses near the noise floor → ratio γ/sill→1
          → SE inversion l^2 ~ -h^2/(2 ln(1-ratio)) blows up
        * No goodness-of-fit check on the ellipse regression
        * No saturation flag — clipped values returned silently as "ok"
      This caused NS planners to over-smooth the GP posterior and lose
      to stationary planners on every field (the JD-paper "fixed-
      background reasoning" failure mode in disguise).  Five layered
      guards now reject unreliable fits before they reach the kernel.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(X)
    # P1.1: raised from N<20 to N<40 — a variogram needs spatial spread,
    # not just sample count, and 20 points on a 25x25 m field gives only
    # ~190 pairs which is below the per-sector threshold once filtered.
    if N < 40:
        return 2.0, 2.0, 0.0, {"reason": "too_few_points", "n": N}

    # Subtract empirical mean (reduces bias in variogram)
    r = y - y.mean()

    if rng is None:
        rng = np.random.default_rng()

    I, J = np.triu_indices(N, k=1)
    if len(I) > max_pairs:
        # P1.7: rng is now caller-controlled (was hardcoded seed=0).
        sel = rng.choice(len(I), size=max_pairs, replace=False)
        I, J = I[sel], J[sel]

    D = X[J] - X[I]
    G = 0.5 * (r[I] - r[J]) ** 2   # empirical half squared differences
    lag = np.linalg.norm(D, axis=1)
    ang = np.arctan2(D[:, 1], D[:, 0])  # (-pi, pi]
    # Fold to [0, pi) (variogram is symmetric about origin)
    ang_folded = np.mod(ang, np.pi)

    # Restrict to lags within [0.3, 10] m (small: noisy; large: underconstrained)
    mask = (lag >= 0.3) & (lag <= 10.0)
    D, G, lag, ang_folded = D[mask], G[mask], lag[mask], ang_folded[mask]
    # P1.1: raised from 50 to 200 — sparse-bin estimates with <200 pairs
    # are too noisy to support 8-sector binning.
    if len(D) < 200:
        return 2.0, 2.0, 0.0, {"reason": "too_few_usable_pairs",
                                 "pairs_kept": int(len(D))}

    # Fit sill (total variance) and lengthscale per angular sector
    sector_edges = np.linspace(0, np.pi, n_angles + 1)
    l_per_angle = []           # effective lengthscale per sector centre
    sill_collapsed_count = 0
    sector_saturated_count = 0
    for k in range(n_angles):
        lo, hi = sector_edges[k], sector_edges[k + 1]
        sel = (ang_folded >= lo) & (ang_folded < hi)
        # P1.1: raised from 20 to 30.
        if sel.sum() < 30:
            continue
        h = lag[sel]
        g = G[sel]
        # Bin by lag, average
        n_bins = 6
        bins = np.linspace(0.3, 10.0, n_bins + 1)
        h_centres, g_means = [], []
        for b in range(n_bins):
            in_bin = (h >= bins[b]) & (h < bins[b + 1])
            if in_bin.sum() >= 3:
                h_centres.append(0.5 * (bins[b] + bins[b + 1]))
                g_means.append(g[in_bin].mean())
        # P1.1: raised from 3 to 4 — need >3 to fit a meaningful SE.
        if len(h_centres) < 4:
            continue
        h_c = np.asarray(h_centres)
        g_c = np.asarray(g_means) - noise_var       # subtract nugget
        g_c = np.clip(g_c, 1e-3, None)
        sill = max(g_c.max(), 1e-3)
        # P1.2: variogram sill collapse detection.  When the empirical
        # variance after nugget subtraction barely exceeds the noise
        # floor, the SE inverse formula's ratio γ/sill saturates near 1
        # and the lengthscale estimate diverges.  Skip the sector — it
        # carries no usable signal.
        if sill < 0.5 * noise_var:
            sill_collapsed_count += 1
            continue
        # g_c / sill ~ 1 - exp(-h^2/(2 l^2))  =>  exp(-h^2/(2 l^2)) ~ 1 - g_c/sill
        # -> l^2 ~ -h^2 / (2 ln(1 - g_c/sill))
        ratio = np.clip(g_c / sill, 1e-3, 0.98)
        ln_term = -2.0 * np.log(1.0 - ratio)
        l_sq_estimates = (h_c ** 2) / np.maximum(ln_term, 1e-4)
        l_sector = float(np.sqrt(np.median(l_sq_estimates)))
        # P1.3: per-sector saturation rejection.  If even the per-sector
        # estimate hits l_max, the SE inverse has saturated — discard
        # this sector instead of letting it vote in the ellipse fit.
        if l_sector >= 0.95 * l_max:
            sector_saturated_count += 1
            continue
        theta_c = 0.5 * (lo + hi)
        l_per_angle.append((theta_c, l_sector, sector_edges[k], sector_edges[k+1]))

    # P1.4: raised from 3 to 5 — with 8 sectors total, require a majority
    # to vote in the ellipse fit, otherwise the fit is underdetermined
    # and the orientation is essentially random.
    if len(l_per_angle) < 5:
        return 2.0, 2.0, 0.0, {"reason": "too_few_sectors",
                                 "n_sectors": len(l_per_angle),
                                 "sill_collapsed": sill_collapsed_count,
                                 "sector_saturated": sector_saturated_count}

    thetas = np.array([t for (t, _, _, _) in l_per_angle])
    ls = np.array([l for (_, l, _, _) in l_per_angle])
    ls = np.clip(ls, 0.1, 20.0)

    # Fit ellipse: 1/l(theta)^2 = A cos^2(theta - theta_p) + B sin^2(theta - theta_p)
    #   => 1/l^2 = (A+B)/2 + (A-B)/2 cos(2(theta - theta_p))
    # Regress y := 1/l^2 on [1, cos(2 theta), sin(2 theta)]
    y_reg = 1.0 / (ls ** 2)
    F = np.column_stack([np.ones_like(thetas),
                         np.cos(2 * thetas), np.sin(2 * thetas)])
    coef, *_ = np.linalg.lstsq(F, y_reg, rcond=None)
    c0, c1, c2 = coef
    # (A+B)/2 = c0; amplitude R = sqrt(c1^2 + c2^2); theta_p = 0.5 atan2(c2, c1)
    R = float(np.hypot(c1, c2))
    A = max(c0 + R, 1e-6)
    B = max(c0 - R, 1e-6)
    # A corresponds to 1/l_short^2, B to 1/l_long^2 (since A >= B)
    l_short = float(1.0 / np.sqrt(A))
    l_long  = float(1.0 / np.sqrt(B))
    # theta_short = direction where 1/l^2 is MAX = short axis
    theta_short = 0.5 * float(np.arctan2(c2, c1))
    # theta = long-axis direction = theta_short + pi/2
    theta = theta_short + np.pi / 2.0
    # Bring to [-pi/2, pi/2]
    while theta > np.pi / 2:
        theta -= np.pi
    while theta < -np.pi / 2:
        theta += np.pi

    # P1.5: ellipse goodness-of-fit residual check.  Compare per-sector
    # l estimates to the values predicted by the fitted ellipse; if the
    # relative RMS residual exceeds 30%, the ellipse is fitting noise
    # rather than a real anisotropic field structure.  Reject the fit.
    pred_inv_l_sq = c0 + c1 * np.cos(2 * thetas) + c2 * np.sin(2 * thetas)
    pred_inv_l_sq = np.clip(pred_inv_l_sq, 1e-6, None)
    pred_l = 1.0 / np.sqrt(pred_inv_l_sq)
    mean_l = float(np.mean(ls))
    rel_residual = float(
        np.sqrt(np.mean((ls - pred_l) ** 2)) / max(mean_l, 1e-6))
    if rel_residual > 0.30:
        return 2.0, 2.0, 0.0, {"reason": "ellipse_fit_poor",
                                "rel_residual": rel_residual,
                                "n_sectors_used": len(l_per_angle),
                                "sill_collapsed": sill_collapsed_count,
                                "sector_saturated": sector_saturated_count}

    # P1.6: saturation flag — if the un-clipped principal axes are at or
    # above l_max, the SE inversion or the ellipse fit hit the ceiling
    # and writing 'l_max' to the kernel would mask divergence.  Reject.
    if l_long >= l_max * 0.99 or l_short >= l_max * 0.99:
        return 2.0, 2.0, 0.0, {"reason": "lengthscale_saturated",
                                "l_long_raw": float(l_long),
                                "l_short_raw": float(l_short),
                                "n_sectors_used": len(l_per_angle),
                                "rel_residual": rel_residual,
                                "sill_collapsed": sill_collapsed_count,
                                "sector_saturated": sector_saturated_count}

    # By convention l1 = long axis (principal), l2 = short axis
    l1 = float(np.clip(l_long,  l_min, l_max))
    l2 = float(np.clip(l_short, l_min, l_max))

    info = {
        "reason": "ok",
        "n_pairs": int(len(D)),
        "n_sectors": len(l_per_angle),
        "per_sector_lengthscales": [float(l) for l in ls],
        "sector_centres_deg": [float(np.degrees(t)) for t in thetas],
        "l1_raw": float(l_long),
        "l2_raw": float(l_short),
        "l1": l1, "l2": l2,
        "theta_rad": theta, "theta_deg": float(np.degrees(theta)),
        "aniso_ratio": float(l1 / l2),
        "rel_residual": rel_residual,
        "sill_collapsed": sill_collapsed_count,
        "sector_saturated": sector_saturated_count,
    }
    return l1, l2, theta, info


# ----------------------------------------------------------------------------
# Applying the result back into a GibbsKernel
# ----------------------------------------------------------------------------
def _realised_at_centre(gk):
    """Read the kernel's currently-realised (l1, l2, theta) at domain centre."""
    import torch
    centres = gk.basis_centers
    dmin = float(centres.min().item())
    dmax = float(centres.max().item())
    x_c = torch.tensor([[0.5 * (dmin + dmax)] * 2],
                       dtype=centres.dtype, device=centres.device)
    with torch.no_grad():
        l1, l2 = gk._lengthscales_at(x_c)
        th = gk._theta_at(x_c)
    return float(l1.item()), float(l2.item()), float(th.item())


def write_into_gibbs_kernel(gk, l1: float, l2: float, theta: float,
                              max_rate: float = 1.5,
                              max_theta_step: float = np.pi / 6.0,
                              ) -> dict:
    """Set the GibbsKernel's basis weights to yield constant (l1, l2, theta).

    The GibbsKernel parametrises l1(x), l2(x), theta(x) as sigmoid of a linear
    combination of RBF basis functions:
        l_i(x) = l_min + (l_max - l_min) * sigmoid( sum_k w_k * phi_k(x) )
    To saturate l_i(x) to a constant target l_target uniformly, every weight
    must be set such that  sum_k w_k * phi_k(x_centre) = logit(t_target),
    where t_target = (l_target - l_min) / (l_max - l_min).

    Since all weights take the same value w, this becomes
        w * phi_sum = logit(t_target)
    where phi_sum = sum_k phi_k(x_centre) is the actual sum of all basis
    activations at the domain centre. This MUST match the GibbsKernel's
    own __init__ convention (which already uses phi_sum, not sqrt(n_basis)).

    P2 hardening (rate cap):
      A single noisy fit can otherwise jump l1 from 2.0 to 4.9 in one
      step (~2.5x).  The kernel then over-smooths the GP posterior and
      the planner cannot recover.  This cap limits the realised
      lengthscale to at most ``max_rate`` × the previous value per
      update; multiple sequential fits compound to the target if the
      data really supports it, but a single noisy fit cannot push the
      kernel into saturation.
    Theta is similarly capped at ``max_theta_step`` per update (default
    30 degrees) using the shortest circular-distance step.

    Returns:
        info dict with the previous, requested, and applied (l1, l2,
        theta) for diagnostic logging.
    """
    import torch

    def inv_sigmoid(v):
        v = float(np.clip(v, 1e-4, 1 - 1e-4))
        return float(np.log(v / (1.0 - v)))

    l_min = gk.l_min
    l_max = gk.l_max

    # P2.1: read previously-realised values, then clip the requested
    # update so neither lengthscale moves by more than max_rate per call.
    l1_prev, l2_prev, th_prev = _realised_at_centre(gk)
    l1_target = float(np.clip(l1, l1_prev / max_rate, l1_prev * max_rate))
    l2_target = float(np.clip(l2, l2_prev / max_rate, l2_prev * max_rate))
    # Re-clip into [l_min, l_max] in case max_rate would push past them.
    l1_target = float(np.clip(l1_target, l_min, l_max))
    l2_target = float(np.clip(l2_target, l_min, l_max))
    # Theta cap (shortest circular distance, in (-pi/2, pi/2)).
    dtheta = theta - th_prev
    while dtheta > np.pi / 2:
        dtheta -= np.pi
    while dtheta < -np.pi / 2:
        dtheta += np.pi
    dtheta = float(np.clip(dtheta, -max_theta_step, max_theta_step))
    theta_target = th_prev + dtheta

    # Recompute phi_sum from the kernel's own buffers — same formula
    # GibbsKernel.__init__ uses, so write-back is consistent with the
    # initial state when l_init is set.
    centers = gk.basis_centers  # (n_basis, 2)
    domain_min = float(centers.min().item())
    domain_max = float(centers.max().item())
    centre_pt = torch.tensor([[0.5 * (domain_min + domain_max)] * 2],
                             dtype=centers.dtype, device=centers.device)
    diffs = centre_pt - centers
    sq_dists = (diffs ** 2).sum(dim=1)
    phi_at_centre = torch.exp(-sq_dists / (2 * gk.basis_sigma_sq))
    phi_sum = float(phi_at_centre.sum().item())

    w1 = inv_sigmoid((l1_target - l_min) / (l_max - l_min)) / phi_sum
    w2 = inv_sigmoid((l2_target - l_min) / (l_max - l_min)) / phi_sum
    # theta sigmoid maps to (-pi/2, pi/2) so we invert (theta + pi/2) / pi
    theta_norm = (theta_target + np.pi / 2) / np.pi
    wt = inv_sigmoid(theta_norm) / phi_sum

    with torch.no_grad():
        gk.basis_weights_l1.fill_(w1)
        gk.basis_weights_l2.fill_(w2)
        gk.basis_weights_theta.fill_(wt)

    return {
        "l1_prev": l1_prev, "l2_prev": l2_prev, "theta_prev": th_prev,
        "l1_requested": float(l1), "l2_requested": float(l2),
        "theta_requested": float(theta),
        "l1_applied": l1_target, "l2_applied": l2_target,
        "theta_applied": theta_target,
        "max_rate": float(max_rate),
        "rate_capped_l1": abs(l1_target - l1) > 1e-6,
        "rate_capped_l2": abs(l2_target - l2) > 1e-6,
        "rate_capped_theta": abs(theta_target - theta) > 1e-6,
    }


def update_kernel_from_data(gk, X: np.ndarray, y: np.ndarray,
                             max_rate: float = 1.5,
                             max_theta_step: float = np.pi / 6.0,
                             **fit_kwargs) -> dict:
    """One-shot spectral update: fit anisotropy to data, write into kernel.

    The writeback respects ``max_rate`` (per-call lengthscale change cap)
    and ``max_theta_step`` (per-call orientation step cap).  When the fit
    is rejected (sparse data, sill collapse, ellipse residual, saturation),
    the kernel is left untouched and the returned info dict carries the
    rejection reason — the caller logs it and moves on.
    """
    l1, l2, theta, info = fit_anisotropy(X, y, **fit_kwargs)
    if info.get("reason") == "ok":
        write_info = write_into_gibbs_kernel(
            gk, l1, l2, theta,
            max_rate=max_rate, max_theta_step=max_theta_step)
        info.update(write_info)
        info["accepted"] = True
    else:
        info["accepted"] = False
    return info


# ----------------------------------------------------------------------------
# MML-uniform: fit global (l1, l2, theta) by marginal likelihood
# ----------------------------------------------------------------------------
def fit_anisotropy_mml(gp_model, n_steps: int = 50,
                        tau_sq: float = 2.0) -> dict:
    """Fit globally-uniform anisotropic (l1, l2, theta) by maximising GP MLL.

    Method: projected-gradient Adam (Williams & Rasmussen 2006, SS5.4).
    After each Adam step the basis weights are projected to be uniform
    (all weights = their mean), enforcing the globally-constant constraint.
    With uniform weights, all 25 w_k share the same gradient (by symmetry
    at the projection point), so the projection does not bias the descent.

    A log-normal prior on l1, l2 prevents the l->l_max feedback loop
    (Gelman et al. 2006, Bayesian Analysis 1(3):515-533):
        penalty = sum_i (log l_i - log l_init)^2 / (2*tau_sq)

    Works correctly with N>=10 samples regardless of domain/range ratio,
    unlike the directional variogram which requires >=150 samples and a
    domain >=10x the correlation range (Journel & Huijbregts 1978).

    Args:
        gp_model: NonstationaryGPModel with training data loaded.
        n_steps: Adam gradient steps.
        tau_sq: Log-normal prior variance on lengthscales. tau_sq=2.0 gives
            a 2-sigma range of exp(+/-2*sqrt(2)) ~ [l_init/17, 17*l_init].

    Returns:
        info dict (same schema as fit_anisotropy / update_kernel_from_data).
    """
    import math
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    gk = gp_model.gibbs_kernel
    l_init = float(gp_model.l_init)
    mu_prior = math.log(l_init)

    # Update learned mean from sample mean before fixing it.
    # MLL then captures kernel-shape effects only (not mean offset).
    y_np = gp_model.train_y.cpu().numpy()
    gp_model._learned_mean = float(y_np.mean())
    gp_model._rebuild_model()

    gp_model.model.train()
    gp_model.likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': [gk.basis_weights_l1],    'lr': 0.05},
        {'params': [gk.basis_weights_l2],    'lr': 0.05},
        {'params': [gk.basis_weights_theta], 'lr': 0.02},
        {'params': [gk._log_signal_var],     'lr': 0.05},
        {'params': [gp_model.model.mean_module.constant], 'lr': 0.1},
    ])
    mll_fn = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model.model)

    # Single centre point for prior evaluation.
    # Using all K=25 basis centres would multiply the prior strength by K,
    # overwhelming the MLL signal (MLL improvement ~ 0.4, K*prior ~ 5).
    centres = gk.basis_centers
    dmin = float(centres.min().item())
    dmax = float(centres.max().item())
    centre_pt = torch.tensor([[0.5 * (dmin + dmax)] * 2],
                              dtype=centres.dtype, device=centres.device)
    best_loss = float('inf')

    for _ in range(n_steps):
        optimizer.zero_grad()
        out = gp_model.model(gp_model.train_x)
        loss = -mll_fn(out, gp_model.train_y)

        # Log-normal prior at domain centre: (log l_i - log l_init)^2 / (2*tau_sq).
        # Evaluated at ONE point so the prior is weakly informative relative
        # to the MLL improvement from learning anisotropy (Gelman et al. 2006).
        l1_c, l2_c = gk._lengthscales_at(centre_pt)
        prior_penalty = (
            (torch.log(l1_c) - mu_prior).pow(2) +
            (torch.log(l2_c) - mu_prior).pow(2)
        ).sum() / (2.0 * tau_sq)
        loss = loss + prior_penalty

        loss.backward()
        optimizer.step()

        # Project to uniform: collapse all weights to their mean.
        # This enforces the globally-constant (l1, l2, theta) constraint.
        with torch.no_grad():
            gk.basis_weights_l1.fill_(gk.basis_weights_l1.mean().item())
            gk.basis_weights_l2.fill_(gk.basis_weights_l2.mean().item())
            gk.basis_weights_theta.fill_(gk.basis_weights_theta.mean().item())

        if loss.item() < best_loss:
            best_loss = loss.item()

    gp_model.model.eval()
    gp_model.likelihood.eval()

    # Persist mean learned during optimisation.
    gp_model._learned_mean = float(
        gp_model.model.mean_module.constant.item())

    # Read the optimised uniform values at domain centre.
    l1_raw, l2_raw, theta_raw = _realised_at_centre(gk)

    # Final write WITH rate cap for planner stability.
    write_info = write_into_gibbs_kernel(
        gk, l1_raw, l2_raw, theta_raw, max_rate=1.5)
    gp_model._rebuild_model()

    l1_a = write_info['l1_applied']
    l2_a = write_info['l2_applied']
    return {
        'reason': 'ok',
        'accepted': True,
        'method': 'mml_uniform',
        'n_steps': n_steps,
        'l1_raw': l1_raw,
        'l2_raw': l2_raw,
        'theta_raw': theta_raw,
        'l1_applied': l1_a,
        'l2_applied': l2_a,
        'theta_applied': write_info['theta_applied'],
        'aniso_ratio': l1_a / max(l2_a, 1e-6),
        'mll_best': -best_loss,
        'rate_capped_l1': write_info['rate_capped_l1'],
        'rate_capped_l2': write_info['rate_capped_l2'],
        'rate_capped_theta': write_info['rate_capped_theta'],
    }
