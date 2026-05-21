"""
Anisotropic Paciorek Non-Stationary Kernel for GPyTorch

Implements the Paciorek & Schervish (2004) nonstationary kernel (Theorem 1)
for D=2 with full anisotropic local covariance matrices:

    Sigma(x) = R(theta) @ diag(l1^2, l2^2) @ R(theta)^T

where l1(x), l2(x) are spatially varying lengthscales and theta(x) is a
spatially varying rotation angle, all parameterized by RBF basis functions
on a coarse grid.

Kernel formula (Paciorek Theorem 1):
    k(x, x') = sigma_f^2
              * |Sigma_x|^{1/4} * |Sigma_{x'}|^{1/4} * |Sigma_avg|^{-1/2}
              * exp(-Q/2)

where:
    Sigma_avg = (Sigma_x + Sigma_{x'}) / 2
    Q = d^T @ Sigma_avg^{-1} @ d
    d = x - x'

Stationary limit: when l1=l2=l and theta=0, Sigma(x) = l^2*I for all x,
    k(x,x') = sigma_f^2 * exp(-||d||^2 / (2*l^2))
which matches GPyTorch's ScaleKernel(RBFKernel()) with the same l.

Parameters: 3 basis weight fields on grid_size^2 grid = 3*grid_size^2 learnable:
- basis_weights_l1 (grid_size^2) -- log-lengthscale field 1
- basis_weights_l2 (grid_size^2) -- log-lengthscale field 2
- basis_weights_theta (grid_size^2) -- rotation angle field
Plus _log_signal_var (1) for overall amplitude.

Reference: Paciorek & Schervish (2004) NIPS, Theorem 1; Gibbs (1997)
Note: cite as 2004 (NIPS proceedings), not 2006 (different review paper).
"""

import torch
import torch.nn as nn
from gpytorch.kernels import Kernel
import math


class GibbsKernel(Kernel):
    """
    Non-stationary anisotropic Paciorek kernel with learned spatially varying
    lengthscales l1(x), l2(x) and rotation angle theta(x).

    The fields are computed from RBF basis functions centered on a coarse grid.
    The basis weights are learnable parameters optimized via GP marginal likelihood.
    """

    has_lengthscale = False  # We handle lengthscale internally

    def __init__(self, grid_size=5, domain_min=0.0, domain_max=25.0,
                 l_min=0.5, l_max=5.0, l_init=2.0, signal_var=1.0,
                 **kwargs):
        """
        Args:
            grid_size: Number of basis centers per dimension (grid_size^2 total)
            domain_min: Domain lower bound (meters)
            domain_max: Domain upper bound (meters)
            l_min: Minimum lengthscale (meters)
            l_max: Soft upper bound for lengthscale (meters)
            l_init: Initial lengthscale (uniform, meters)
            signal_var: Signal variance (outputscale)
        """
        super().__init__(**kwargs)

        self.grid_size = grid_size
        self.l_min = l_min
        self.l_max = l_max

        # Signal variance as learnable parameter (log-space for positivity)
        self._log_signal_var = nn.Parameter(torch.tensor(math.log(signal_var)))

        # Create basis function centers on a grid
        centers_1d = torch.linspace(domain_min, domain_max, grid_size)
        cx, cy = torch.meshgrid(centers_1d, centers_1d, indexing='xy')
        self.register_buffer('basis_centers', torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1))
        # basis_centers: (grid_size^2, 2)

        n_basis = grid_size * grid_size

        # Basis function width: cover the domain with overlap
        spacing = (domain_max - domain_min) / (grid_size - 1) if grid_size > 1 else (domain_max - domain_min)
        self.register_buffer('basis_sigma_sq', torch.tensor(spacing ** 2))

        # Compute phi_sum at domain center for initialization calibration
        center_pt = torch.tensor([[0.5 * (domain_min + domain_max)] * 2])
        diffs = center_pt - self.basis_centers
        sq_dists = (diffs ** 2).sum(dim=1)
        phi_at_center = torch.exp(-sq_dists / (2 * self.basis_sigma_sq))
        phi_sum = phi_at_center.sum().item()

        # Initialize l1 and l2 so that l1(x) = l2(x) = l_init everywhere
        t = (l_init - l_min) / (l_max - l_min)
        t = max(0.01, min(0.99, t))
        logit_t = math.log(t / (1.0 - t))
        init_val = logit_t / phi_sum

        self.basis_weights_l1 = nn.Parameter(torch.full((n_basis,), init_val))
        self.basis_weights_l2 = nn.Parameter(torch.full((n_basis,), init_val))

        # Initialize theta so that theta(x) = 0 everywhere
        # sigmoid(0) = 0.5, so 0.5 * pi = pi/2... we want 0.
        # Use: theta = (sigmoid(w) - 0.5) * pi -> range (-pi/2, pi/2)
        # At w=0: theta=0. So init weights to 0.
        # But with overlapping basis functions, init_val should account for phi_sum
        # sigmoid(phi_sum * w) - 0.5 = 0 => w = 0
        self.basis_weights_theta = nn.Parameter(torch.zeros(n_basis))

    @property
    def signal_var(self):
        """Current signal variance (learnable, positive via exp)."""
        return torch.exp(self._log_signal_var).item()

    def _basis_activations(self, x):
        """
        Compute basis function activations at locations x.

        Args:
            x: (N, 2) input locations

        Returns:
            phi: (N, K) basis activations
        """
        diffs = x.unsqueeze(1) - self.basis_centers.unsqueeze(0)  # (N, K, 2)
        sq_dists = (diffs ** 2).sum(dim=2)  # (N, K)
        phi = torch.exp(-sq_dists / (2 * self.basis_sigma_sq))  # (N, K)
        return phi

    def _lengthscales_at(self, X):
        """
        Compute spatially varying lengthscales l1(x), l2(x) at each point.

        l_i(x) = l_min + (l_max - l_min) * sigmoid(sum_k a_k * phi_k(x))

        Args:
            X: (N, 2) input locations

        Returns:
            l1, l2: both (N,) lengthscales at each location
        """
        phi = self._basis_activations(X)  # (N, K)

        weighted_l1 = phi @ self.basis_weights_l1  # (N,)
        weighted_l2 = phi @ self.basis_weights_l2  # (N,)

        l1 = self.l_min + (self.l_max - self.l_min) * torch.sigmoid(weighted_l1)
        l2 = self.l_min + (self.l_max - self.l_min) * torch.sigmoid(weighted_l2)

        return l1, l2

    def _theta_at(self, X):
        """
        Compute spatially varying rotation angle theta(x) at each point.

        theta(x) = (sigmoid(sum_k a_k * phi_k(x)) - 0.5) * pi
        Range: (-pi/2, pi/2)

        Args:
            X: (N, 2) input locations

        Returns:
            theta: (N,) rotation angle at each location
        """
        phi = self._basis_activations(X)  # (N, K)
        weighted = phi @ self.basis_weights_theta  # (N,)
        theta = (torch.sigmoid(weighted) - 0.5) * math.pi
        return theta

    def covariance_at(self, X):
        """
        Compute local 2x2 anisotropic covariance matrices at each point.

        Sigma(x) = R(theta) @ diag(l1^2, l2^2) @ R(theta)^T

        Closed-form (no matrix multiply needed):
            Sigma_00 = l1^2 * cos^2(theta) + l2^2 * sin^2(theta)
            Sigma_01 = (l1^2 - l2^2) * cos(theta) * sin(theta)
            Sigma_11 = l1^2 * sin^2(theta) + l2^2 * cos^2(theta)

        Args:
            X: (N, 2) input locations

        Returns:
            Sigma: (N, 2, 2) local covariance matrices
        """
        l1, l2 = self._lengthscales_at(X)
        theta = self._theta_at(X)

        l1_sq = l1 ** 2
        l2_sq = l2 ** 2
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        cos_t_sq = cos_t ** 2
        sin_t_sq = sin_t ** 2
        cos_sin = cos_t * sin_t

        S00 = l1_sq * cos_t_sq + l2_sq * sin_t_sq
        S01 = (l1_sq - l2_sq) * cos_sin
        S11 = l1_sq * sin_t_sq + l2_sq * cos_t_sq

        # Stack into (N, 2, 2)
        Sigma = torch.stack([
            torch.stack([S00, S01], dim=-1),
            torch.stack([S01, S11], dim=-1)
        ], dim=-2)

        return Sigma

    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the anisotropic Paciorek nonstationary kernel matrix.

        k(x, x') = sigma_f^2
                  * |Sigma_x|^{1/4} * |Sigma_{x'}|^{1/4} * |Sigma_avg|^{-1/2}
                  * exp(-Q/2)

        where:
            Sigma_avg = (Sigma_x + Sigma_{x'}) / 2
            Q = d^T @ Sigma_avg^{-1} @ d
            d = x - x'

        All computed in closed form for 2x2 matrices.

        Args:
            x1: (N, 2) first set of points
            x2: (M, 2) second set of points
            diag: If True, return only diagonal elements

        Returns:
            K: (N, M) kernel matrix, or (N,) if diag=True
        """
        signal_var = torch.exp(self._log_signal_var)

        # Get local covariance matrices
        Sigma1 = self.covariance_at(x1)  # (N, 2, 2)
        Sigma2 = self.covariance_at(x2)  # (M, 2, 2)

        # Determinants of Sigma: det(2x2) = a*d - b*c
        # |Sigma| = S00*S11 - S01^2  (since S01 = S10)
        det1 = Sigma1[:, 0, 0] * Sigma1[:, 1, 1] - Sigma1[:, 0, 1] ** 2  # (N,)
        det2 = Sigma2[:, 0, 0] * Sigma2[:, 1, 1] - Sigma2[:, 0, 1] ** 2  # (M,)

        if diag:
            # Diagonal: x1[i] with x2[i], same size
            N = x1.shape[0]

            # Sigma_avg = (Sigma1 + Sigma2) / 2
            Sigma_avg = (Sigma1 + Sigma2) / 2  # (N, 2, 2)

            # Add jitter for numerical stability
            jitter = 1e-6
            Sigma_avg = Sigma_avg.clone()
            Sigma_avg[:, 0, 0] = Sigma_avg[:, 0, 0] + jitter
            Sigma_avg[:, 1, 1] = Sigma_avg[:, 1, 1] + jitter

            # det(Sigma_avg)
            det_avg = Sigma_avg[:, 0, 0] * Sigma_avg[:, 1, 1] - Sigma_avg[:, 0, 1] ** 2  # (N,)

            # Prefactor: |Sigma_x|^{1/4} * |Sigma_{x'}|^{1/4} * |Sigma_avg|^{-1/2}
            # = (|Sigma_x| * |Sigma_{x'}|)^{1/4} / |Sigma_avg|^{1/2}
            prefactor = (det1 * det2).clamp(min=1e-30).pow(0.25) / det_avg.clamp(min=1e-30).pow(0.5)

            # Displacement d = x1 - x2
            d = x1 - x2  # (N, 2)

            # Inverse of 2x2: [[d, -b], [-c, a]] / det
            inv_avg_00 = Sigma_avg[:, 1, 1] / det_avg
            inv_avg_01 = -Sigma_avg[:, 0, 1] / det_avg
            inv_avg_11 = Sigma_avg[:, 0, 0] / det_avg

            # Q = d^T @ Sigma_avg^{-1} @ d
            Q = (d[:, 0] ** 2 * inv_avg_00 +
                 2 * d[:, 0] * d[:, 1] * inv_avg_01 +
                 d[:, 1] ** 2 * inv_avg_11)

            K_diag = signal_var * prefactor * torch.exp(-Q / 2)
            return K_diag

        else:
            # Full matrix: (N, M)
            N = x1.shape[0]
            M = x2.shape[0]

            # Expand Sigma1 and Sigma2 for broadcasting: (N, 1, 2, 2) and (1, M, 2, 2)
            S1 = Sigma1.unsqueeze(1)  # (N, 1, 2, 2)
            S2 = Sigma2.unsqueeze(0)  # (1, M, 2, 2)

            # Sigma_avg = (S1 + S2) / 2  -> (N, M, 2, 2)
            Sigma_avg = (S1 + S2) / 2

            # Add jitter
            jitter = 1e-6
            Sigma_avg = Sigma_avg.clone()
            Sigma_avg[:, :, 0, 0] = Sigma_avg[:, :, 0, 0] + jitter
            Sigma_avg[:, :, 1, 1] = Sigma_avg[:, :, 1, 1] + jitter

            # det(Sigma_avg): (N, M)
            det_avg = (Sigma_avg[:, :, 0, 0] * Sigma_avg[:, :, 1, 1] -
                       Sigma_avg[:, :, 0, 1] ** 2)

            # Prefactor: (N, M)
            det1_exp = det1.unsqueeze(1)  # (N, 1)
            det2_exp = det2.unsqueeze(0)  # (1, M)
            prefactor = ((det1_exp * det2_exp).clamp(min=1e-30).pow(0.25) /
                         det_avg.clamp(min=1e-30).pow(0.5))

            # Displacement: d[i,j] = x1[i] - x2[j]
            d = x1.unsqueeze(1) - x2.unsqueeze(0)  # (N, M, 2)

            # Inverse of 2x2 Sigma_avg
            inv_avg_00 = Sigma_avg[:, :, 1, 1] / det_avg  # (N, M)
            inv_avg_01 = -Sigma_avg[:, :, 0, 1] / det_avg
            inv_avg_11 = Sigma_avg[:, :, 0, 0] / det_avg

            # Q = d^T @ inv(Sigma_avg) @ d
            Q = (d[:, :, 0] ** 2 * inv_avg_00 +
                 2 * d[:, :, 0] * d[:, :, 1] * inv_avg_01 +
                 d[:, :, 1] ** 2 * inv_avg_11)

            K = signal_var * prefactor * torch.exp(-Q / 2)
            return K

    def get_lengthscale_field(self, resolution=0.5, domain_min=0.0, domain_max=25.0):
        """
        Get current kernel fields on a grid for visualization.

        Returns:
            X, Y: meshgrid arrays
            L1: lengthscale-1 values on grid
            L2: lengthscale-2 values on grid
            Theta: rotation angle values on grid
        """
        with torch.no_grad():
            x = torch.arange(domain_min, domain_max + 1e-9, resolution)
            y = torch.arange(domain_min, domain_max + 1e-9, resolution)
            X, Y = torch.meshgrid(x, y, indexing='xy')
            grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
            grid = grid.to(self.basis_centers.device)

            l1, l2 = self._lengthscales_at(grid)
            theta = self._theta_at(grid)

            L1 = l1.reshape(X.shape)
            L2 = l2.reshape(X.shape)
            Theta = theta.reshape(X.shape)

        return X.numpy(), Y.numpy(), L1.cpu().numpy(), L2.cpu().numpy(), Theta.cpu().numpy()
