"""
GP Model with Gibbs Non-Stationary Kernel

Wraps the anisotropic Paciorek kernel into a GPyTorch ExactGP model with the
same interface as info_gain.gp_model.GPModel, plus:
- Periodic hyperparameter optimization via marginal likelihood
- Lengthscale field snapshots for visualization

The key difference from the stationary GP:
- The kernel has spatially varying anisotropic covariance Sigma(x)
- Three fields: l1(x), l2(x) (lengthscales) and theta(x) (rotation)
- The parameters are learned online from data
- This causes the GP to allocate different effective resolution
  across the domain, concentrating samples near complex regions
"""

import math
import torch
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch

from .gibbs_kernel import GibbsKernel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_device(device_arg):
    """Resolve a user-supplied device argument to a torch.device.

    Accepts None, "auto", "cpu", "cuda", "cuda:N", or a torch.device.
    "auto" / None  -> cuda if available, else cpu.
    """
    if isinstance(device_arg, torch.device):
        return device_arg
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class GibbsGPModel(ExactGP):
    """ExactGP with Gibbs non-stationary kernel."""

    def __init__(self, train_x, train_y, likelihood, gibbs_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = gibbs_kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class NonstationaryGPModel:
    """
    GP with anisotropic Paciorek non-stationary kernel.

    Features:
    - Spatially varying anisotropic covariance via Gibbs kernel
    - Three learned fields: l1(x), l2(x), theta(x)
    - Periodic online optimization of kernel parameters
    - Saves lengthscale field snapshots for thesis visualization
    - GPU acceleration

    Interface (compatible with info_gain.gp_model.GPModel):
    - fit(X, y)
    - predict(X_test)
    - add_observation(x_new, y_new)
    - get_training_data()
    - n_observations property

    Additional:
    - optimize_lengthscale(n_steps): Optimize basis weights via marginal likelihood
    - get_lengthscale_field(): Get current l1, l2, theta maps
    - save_lengthscale_snapshot(path, step): Save fields for visualization
    """

    def __init__(self, noise_var=0.36, signal_var=1.0, l_init=2.0,
                 grid_size=5, l_min=0.5, l_max=5.0,
                 optimize_every=10, optimize_steps=20,
                 kernel_mode='smooth_gibbs', device=None):
        """
        Args:
            noise_var: Observation noise variance
            signal_var: Signal variance (outputscale)
            l_init: Initial lengthscale (uniform)
            grid_size: Basis grid size (grid_size^2 basis functions)
            l_min: Minimum lengthscale
            l_max: Maximum lengthscale
            optimize_every: Optimize lengthscale every N observations
            optimize_steps: Gradient steps per optimization
            kernel_mode:
                'smooth_gibbs' (default) — full spatially-varying Paciorek
                  kernel. Adam on all 75 basis weights with:
                  (1) MLL, (2) per-point log-normal prior (prevents global
                  drift), (3) spatial smoothness on the log-lengthscale field
                  (prevents l→l_max feedback in sparse regions). Warm-starts
                  from mml_uniform on the first optimisation call.
                'mml_uniform' — globally uniform (l1,l2,theta); effectively
                  anisotropic stationary. Useful for ablation.
                'spectral' (deprecated) — directional variogram; broken for
                  domains with range/domain ratio > 0.3.
                'gibbs' (legacy) — MAP on 75 basis weights, no smoothness.
            device: torch device for this GP instance. None/"auto" -> cuda if available.
        """
        self.device = _resolve_device(device)
        self.noise_var = noise_var
        self.signal_var = signal_var
        self.l_init = l_init
        self.grid_size = grid_size
        self.l_min = l_min
        self.l_max = l_max
        self.optimize_every = optimize_every
        self.optimize_steps = optimize_steps
        # kernel_mode: 'gibbs' (original MAP on 76 basis weights) or
        #              'spectral' (direct anisotropy fit from empirical
        #              autocorrelation of residuals — JD-motivated).
        self.kernel_mode = kernel_mode

        self.train_x = None
        self.train_y = None
        self.model = None
        self.likelihood = None

        # Create a persistent kernel that carries learned weights across refits
        self.gibbs_kernel = GibbsKernel(
            grid_size=grid_size, l_min=l_min, l_max=l_max,
            l_init=l_init, signal_var=signal_var
        ).to(self.device)

        # 4-connected adjacency list for the grid_size×grid_size basis grid.
        # With indexing='xy': basis_centers[i*G+j] = (x=centers[j], y=centers[i]).
        # Adjacent pairs: (i,j)↔(i,j+1) horizontal, (i,j)↔(i+1,j) vertical.
        G = grid_size
        self._adj_pairs = []
        for i in range(G):
            for j in range(G):
                k = i * G + j
                if j + 1 < G:
                    self._adj_pairs.append((k, k + 1))       # horizontal
                if i + 1 < G:
                    self._adj_pairs.append((k, k + G))       # vertical
        # 40 pairs for 5×5 grid

        # Persistent learned mean (survives model rebuilds)
        self._learned_mean = 0.0

        # Track optimization schedule
        self._obs_since_last_opt = 0
        self._total_opt_count = 0

        # P3: per-trial fit log.  When `set_fit_log_path(path)` is called
        # by the planner, every spectral-fit attempt (including
        # rejections) appends a row capturing the reason, raw vs applied
        # lengthscales, and goodness-of-fit residual.  Empty path = no log.
        self._fit_log_path = None
        self._fit_log_step_fn = None  # optional callable returning current sample step

    def set_fit_log_path(self, path, step_fn=None):
        """Configure per-trial spectral-fit logging.  See P3 in plan."""
        self._fit_log_path = path
        self._fit_log_step_fn = step_fn

    def _append_fit_log(self, info: dict):
        """Append one row to the fit_log.csv set by set_fit_log_path."""
        if self._fit_log_path is None:
            return
        import csv
        import os
        path = self._fit_log_path
        is_new = not os.path.isfile(path)
        # Step is supplied via callback (planner gives current sample idx).
        step = -1
        try:
            if self._fit_log_step_fn is not None:
                step = int(self._fit_log_step_fn())
        except Exception:
            pass
        n_obs = int(self.train_x.shape[0]) if self.train_x is not None else 0
        row = {
            'step': step,
            'n_obs': n_obs,
            'reason': info.get('reason', ''),
            'accepted': bool(info.get('accepted', False)),
            'l1_raw': info.get('l1_raw', float('nan')),
            'l2_raw': info.get('l2_raw', float('nan')),
            'theta_raw': info.get('theta_rad', float('nan')),
            'l1_applied': info.get('l1_applied', float('nan')),
            'l2_applied': info.get('l2_applied', float('nan')),
            'theta_applied': info.get('theta_applied', float('nan')),
            'aniso_ratio': info.get('aniso_ratio', float('nan')),
            'n_sectors_used': info.get('n_sectors',
                                        info.get('n_sectors_used', 0)),
            'rel_residual': info.get('rel_residual', float('nan')),
            'sill_collapsed': info.get('sill_collapsed', 0),
            'sector_saturated': info.get('sector_saturated', 0),
            'rate_capped_l1': bool(info.get('rate_capped_l1', False)),
            'rate_capped_l2': bool(info.get('rate_capped_l2', False)),
            'rate_capped_theta': bool(info.get('rate_capped_theta', False)),
        }
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                w.writeheader()
            w.writerow(row)

    def fit(self, X, y):
        """
        Fit GP to training data.

        Args:
            X: (N, D) training inputs
            y: (N,) training targets
        """
        self.train_x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.train_y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        self._rebuild_model()

    def _rebuild_model(self):
        """Rebuild GP model with current training data and kernel weights."""
        self.likelihood = GaussianLikelihood().to(self.device)
        self.likelihood.noise = self.noise_var

        self.model = GibbsGPModel(
            self.train_x, self.train_y,
            self.likelihood, self.gibbs_kernel
        ).to(self.device)

        # Restore learned mean from previous optimization rounds
        self.model.mean_module.constant.data.fill_(self._learned_mean)

        self.model.eval()
        self.likelihood.eval()

    def predict(self, X_test):
        """
        Predict mean and variance at test points.

        Returns torch tensors on `self.device`.  See `predict_torch` for
        the GPU-acquisition-friendly alias and `get_training_tensors`
        for direct tensor access without the `.cpu().numpy()` round-trip
        that `get_training_data()` does.

        Args:
            X_test: (M, D) test inputs

        Returns:
            mean: (M,) posterior mean (torch tensor on self.device)
            var:  (M,) posterior variance (torch tensor on self.device)
        """
        if self.model is None:
            M = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
            return (torch.zeros(M, device=self.device),
                    torch.ones(M, device=self.device) * self.gibbs_kernel.signal_var)

        if isinstance(X_test, torch.Tensor):
            X_test = X_test.to(device=self.device, dtype=torch.float32)
        else:
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.model(X_test)
            mean = pred.mean
            var = pred.variance

        return mean, var

    def predict_torch(self, X_test):
        """Tensor-in / tensor-out predict for the GPU acquisition layer."""
        return self.predict(X_test)

    def get_training_tensors(self):
        """Return (X_train, y_train) as torch tensors on `self.device`.

        No CPU round-trip.  Use this from the GPU acquisition layer; the
        legacy `get_training_data()` returns numpy and is kept for
        callers that want it.
        """
        return self.train_x, self.train_y

    def add_observation(self, x_new, y_new):
        """
        Add a new observation and refit GP.

        Args:
            x_new: (D,) new input location
            y_new: scalar observation
        """
        x_new = torch.as_tensor(x_new, dtype=torch.float32,
                                 device=self.device).reshape(1, -1)
        y_new = torch.as_tensor([y_new], dtype=torch.float32,
                                 device=self.device)

        if self.train_x is None:
            self.train_x = x_new
            self.train_y = y_new
        else:
            self.train_x = torch.cat([self.train_x, x_new], dim=0)
            self.train_y = torch.cat([self.train_y, y_new], dim=0)

        self._obs_since_last_opt += 1
        self._rebuild_model()

    def should_optimize(self):
        """Check if it's time to optimize lengthscale parameters."""
        return (self._obs_since_last_opt >= self.optimize_every and
                self.train_x is not None and
                len(self.train_x) >= 10)  # Need minimum data

    def optimize_lengthscale(self, n_steps=None, logger=None):
        """
        Optimize GP hyperparameters via marginal likelihood.

        Optimizes:
        - Gibbs kernel basis_weights_l1 (25 params) - lengthscale field 1
        - Gibbs kernel basis_weights_l2 (25 params) - lengthscale field 2
        - Gibbs kernel basis_weights_theta (25 params) - rotation angle field
        - Gibbs kernel signal variance (1 param) - overall kernel amplitude
        - GP mean constant (1 param) - data offset

        Noise variance is kept FIXED (known from sensor spec).

        Args:
            n_steps: Number of gradient steps (default: self.optimize_steps)
            logger: Optional ROS logger for progress messages

        Returns:
            final_loss: Final negative log marginal likelihood
        """
        if self.train_x is None or len(self.train_x) < 10:
            return None

        # --- Smooth-Gibbs branch (default): full spatially-varying kernel ---
        if self.kernel_mode == 'smooth_gibbs':
            return self._optimize_smooth_gibbs(n_steps=n_steps, logger=logger)

        # --- MML-uniform branch ---
        if self.kernel_mode == 'mml_uniform':
            from .spectral_kernel import fit_anisotropy_mml
            info = fit_anisotropy_mml(self, n_steps=80, tau_sq=2.0)
            self._obs_since_last_opt = 0
            self._total_opt_count += 1
            self._append_fit_log(info)
            if logger:
                if info.get('accepted'):
                    capped = []
                    if info.get('rate_capped_l1'): capped.append('l1')
                    if info.get('rate_capped_l2'): capped.append('l2')
                    if info.get('rate_capped_theta'): capped.append('theta')
                    cap_str = (f' [rate-capped: {",".join(capped)}]'
                               if capped else '')
                    logger.info(
                        f'GP mml-uniform (round {self._total_opt_count}): '
                        f'l1={info["l1_applied"]:.2f}, '
                        f'l2={info["l2_applied"]:.2f}, '
                        f'theta={float(info["theta_applied"]*180/3.14159):.1f}deg, '
                        f'aniso={info["aniso_ratio"]:.2f}, '
                        f'mean={self._learned_mean:.2f}{cap_str}')
            return 0.0

        # --- Spectral variogram branch (deprecated — see kernel_mode docstring) ---
        if self.kernel_mode == 'spectral':
            from .spectral_kernel import update_kernel_from_data
            X_np = self.train_x.cpu().numpy()
            y_np = self.train_y.cpu().numpy()
            # Subtract current learned mean so residuals are the right signal
            # Variogram detection needs a wider l_max than the kernel bound
            # so that x-direction sectors on anisotropic fields (e.g.
            # y_compress σ_x=7m) are not rejected by P1.3 (l_sector≥4.75).
            # write_into_gibbs_kernel clips the result back to gk.l_max.
            info = update_kernel_from_data(
                self.gibbs_kernel, X_np, y_np - self._learned_mean,
                signal_var=self.gibbs_kernel.signal_var,
                noise_var=self.noise_var,
                l_min=self.l_min, l_max=self.l_max * 2.5)
            # Also update the learned mean via sample mean (simple closed form)
            self._learned_mean = float(y_np.mean())
            # Rebuild model so the new kernel + mean take effect
            self._rebuild_model()
            self._obs_since_last_opt = 0
            self._total_opt_count += 1
            # P3: append row to per-trial fit_log.csv (no-op if not set).
            self._append_fit_log(info)
            if logger:
                if info.get("reason") == "ok":
                    capped = []
                    if info.get('rate_capped_l1'): capped.append('l1')
                    if info.get('rate_capped_l2'): capped.append('l2')
                    if info.get('rate_capped_theta'): capped.append('theta')
                    cap_str = f' [rate-capped: {",".join(capped)}]' if capped else ''
                    logger.info(
                        f'GP spectral-update (round {self._total_opt_count}): '
                        f'l1={info["l1_applied"]:.2f}, l2={info["l2_applied"]:.2f}, '
                        f'theta={info["theta_deg"]:.1f}deg, '
                        f'aniso={info["aniso_ratio"]:.2f}, '
                        f'mean={self._learned_mean:.2f}, '
                        f'n_pairs={info["n_pairs"]}{cap_str}')
                else:
                    logger.info(f'GP spectral-update skipped: {info["reason"]} '
                                 f'(n_obs={self.train_x.shape[0]})')
            return 0.0

        n_steps = n_steps or self.optimize_steps

        # Put model in training mode for optimization
        self.model.train()
        self.likelihood.train()

        # MAP optimization (Xu & Choi 2011): MLL + log-normal prior on lengthscales.
        # The prior prevents the "l → l_max" feedback loop by penalizing large
        # lengthscales. Without it, MLL always prefers smoother fits, which crushes
        # variance and kills the planner's ability to distinguish explored/unexplored.
        # Noise variance stays fixed (known sensor spec).
        optimizer = torch.optim.Adam([
            {'params': [self.gibbs_kernel.basis_weights_l1], 'lr': 0.05},
            {'params': [self.gibbs_kernel.basis_weights_l2], 'lr': 0.05},
            {'params': [self.gibbs_kernel.basis_weights_theta], 'lr': 0.02},
            {'params': [self.gibbs_kernel._log_signal_var], 'lr': 0.05},
            {'params': [self.model.mean_module.constant], 'lr': 0.1},
        ])
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Prior parameters: log-normal centered on l_init
        # tau_sq controls prior spread: smaller = tighter, larger = more freedom.
        # With K=25 basis points per field (50 terms total), penalty scales as
        # K * (log(l/l_init))^2 / (2*tau_sq). At tau_sq=1.5, doubling the
        # lengthscale costs ~4.0 total — enough to weakly regularize without
        # preventing the kernel from learning useful spatial variation.
        mu_prior = math.log(self.l_init)
        tau_sq = 1.5
        prior_points = self.gibbs_kernel.basis_centers  # (K, 2) fixed grid

        best_loss = float('inf')
        for i in range(n_steps):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)

            # MAP prior: penalize lengthscales deviating from l_init
            l1_p, l2_p = self.gibbs_kernel._lengthscales_at(prior_points)
            prior_penalty = (
                (torch.log(l1_p) - mu_prior).pow(2).sum() +
                (torch.log(l2_p) - mu_prior).pow(2).sum()
            ) / (2 * tau_sq)
            loss = loss + prior_penalty

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val

        # Back to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Persist learned mean for future model rebuilds
        self._learned_mean = self.model.mean_module.constant.item()

        # Reset counter
        self._obs_since_last_opt = 0
        self._total_opt_count += 1

        if logger:
            with torch.no_grad():
                l1, l2 = self.gibbs_kernel._lengthscales_at(self.train_x)
                theta = self.gibbs_kernel._theta_at(self.train_x)
                aniso = l1 / l2
            sig_var = self.gibbs_kernel.signal_var
            learned_mean = self._learned_mean
            logger.info(
                f'GP optimized (round {self._total_opt_count}): '
                f'loss={best_loss:.3f}, '
                f'l1=[{l1.min():.2f},{l1.max():.2f}], '
                f'l2=[{l2.min():.2f},{l2.max():.2f}], '
                f'theta=[{theta.min():.2f},{theta.max():.2f}], '
                f'aniso={aniso.mean():.2f}, '
                f'signal_var={sig_var:.2f}, mean={learned_mean:.2f}'
            )

        return best_loss

    def _optimize_smooth_gibbs(self, n_steps=None, logger=None):
        """Spatially-varying kernel optimisation with smoothness regularisation.

        Three regularisation terms combine with the MLL:
          (1) Per-point log-normal prior: each basis point's l1, l2 pulled
              toward l_init. Prevents global drift to l_max.
              Ref: Gelman et al. (2006) Bayesian Analysis 1(3):515.
          (2) 4-connected spatial smoothness: penalises log-lengthscale
              differences between adjacent basis points on the 5×5 grid.
              Prevents l→l_max in sparse regions (the feedback-loop fix).
          (3) Warm start from mml_uniform on the first call (round 0 only):
              gives a good global anisotropic starting point before local
              adaptation begins. Subsequent calls start from current weights.

        Together these allow the kernel to learn genuine spatial variation:
          - Near field hotspot: shorter l (field varies quickly)
          - At flat domain edges: longer l
          - Across the domain: l1/l2 anisotropy tracks the field direction
        """
        import math as _math
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from .spectral_kernel import fit_anisotropy_mml

        n_steps = n_steps or max(self.optimize_steps, 80)
        gk = self.gibbs_kernel

        # Round 0: warm-start with global anisotropic fit so local adaptation
        # begins from a sensible starting point rather than isotropic l_init.
        if self._total_opt_count == 0:
            fit_anisotropy_mml(self, n_steps=50, tau_sq=2.0)
            # ↑ sets uniform weights to good global (l1, l2, theta)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': [gk.basis_weights_l1],    'lr': 0.05},
            {'params': [gk.basis_weights_l2],    'lr': 0.05},
            {'params': [gk.basis_weights_theta], 'lr': 0.02},
            {'params': [gk._log_signal_var],     'lr': 0.05},
            {'params': [self.model.mean_module.constant], 'lr': 0.1},
        ])
        mll_fn = ExactMarginalLogLikelihood(self.likelihood, self.model)

        mu_prior = _math.log(self.l_init)
        tau_sq   = 2.0    # log-normal prior spread
        # lambda_smooth: weight on spatial smoothness term.
        # Calibrated so that a smooth gradient of log(7/2.5) across the domain
        # (Δlog_l ≈ 0.25 per adjacent pair) costs ~ MLL improvement (~0.4).
        # With 40 pairs: λ × 40 × 0.25² ≈ 0.4  →  λ ≈ 0.16.
        # We use 0.5 for safety margin (prevents runaway, allows variation).
        lambda_smooth = 0.5

        basis_pts = gk.basis_centers          # (K, 2)
        adj = self._adj_pairs                  # list of (i, j) index pairs

        best_loss = float('inf')
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = -mll_fn(self.model(self.train_x), self.train_y)

            l1_b, l2_b = gk._lengthscales_at(basis_pts)   # (K,) each
            log_l1 = torch.log(l1_b)
            log_l2 = torch.log(l2_b)

            # (1) Per-point log-normal prior (normalised by K so it stays
            #     weak relative to the MLL even with 25 basis points).
            per_pt = ((log_l1 - mu_prior).pow(2) +
                       (log_l2 - mu_prior).pow(2)).sum() / (2.0 * tau_sq * len(l1_b))
            loss = loss + per_pt

            # (2) Spatial smoothness: Σ_adjacent (Δlog_l1² + Δlog_l2²).
            smooth = torch.stack([
                (log_l1[i] - log_l1[j]).pow(2) + (log_l2[i] - log_l2[j]).pow(2)
                for i, j in adj
            ]).sum()
            loss = loss + lambda_smooth * smooth

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        self.model.eval()
        self.likelihood.eval()
        self._learned_mean = float(self.model.mean_module.constant.item())
        self._obs_since_last_opt = 0
        self._total_opt_count += 1

        if logger:
            with torch.no_grad():
                l1_all, l2_all = gk._lengthscales_at(basis_pts)
                theta_all = gk._theta_at(basis_pts)
            logger.info(
                f'GP smooth-gibbs (round {self._total_opt_count}): '
                f'l1=[{l1_all.min():.2f},{l1_all.max():.2f}] '
                f'l2=[{l2_all.min():.2f},{l2_all.max():.2f}] '
                f'theta=[{theta_all.min():.2f},{theta_all.max():.2f}] '
                f'aniso_mean={( l1_all/l2_all).mean():.2f} '
                f'mean={self._learned_mean:.2f}')

        # Build info dict for fit_log (same schema as other branches)
        with torch.no_grad():
            l1_c, l2_c = gk._lengthscales_at(basis_pts)
            l1_mean = float(l1_c.mean()); l2_mean = float(l2_c.mean())
        info = {
            'reason': 'ok', 'accepted': True, 'method': 'smooth_gibbs',
            'l1_raw': l1_mean, 'l2_raw': l2_mean,
            'l1_applied': l1_mean, 'l2_applied': l2_mean,
            'theta_applied': 0.0,
            'aniso_ratio': l1_mean / max(l2_mean, 1e-6),
            'mll_best': -best_loss,
            'rate_capped_l1': False, 'rate_capped_l2': False,
            'rate_capped_theta': False,
        }
        self._append_fit_log(info)
        return best_loss

    def get_training_data(self):
        """Return current training data."""
        if self.train_x is None:
            return None, None
        return self.train_x.cpu().numpy(), self.train_y.cpu().numpy()

    @property
    def n_observations(self):
        """Number of observations in training set."""
        return 0 if self.train_x is None else len(self.train_x)

    def get_lengthscale_field(self, resolution=0.5):
        """Get current lengthscale field for visualization."""
        return self.gibbs_kernel.get_lengthscale_field(resolution=resolution)

    def save_lengthscale_snapshot(self, output_dir, step):
        """
        Save lengthscale field snapshot for thesis visualization.

        Args:
            output_dir: Path to save directory
            step: Current sample step number
        """
        X, Y, L1, L2, Theta = self.get_lengthscale_field()
        weights_l1 = self.gibbs_kernel.basis_weights_l1.detach().cpu().numpy()
        weights_l2 = self.gibbs_kernel.basis_weights_l2.detach().cpu().numpy()
        weights_theta = self.gibbs_kernel.basis_weights_theta.detach().cpu().numpy()

        np.savez(
            output_dir / f'lengthscale_step_{step:03d}.npz',
            X=X, Y=Y, L1=L1, L2=L2, Theta=Theta,
            basis_weights_l1=weights_l1,
            basis_weights_l2=weights_l2,
            basis_weights_theta=weights_theta,
            step=step,
            n_observations=self.n_observations
        )
