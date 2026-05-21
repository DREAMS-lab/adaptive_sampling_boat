#!/usr/bin/env python3
"""
Non-Stationary Exact Planner - Gibbs Kernel with Spatially Varying Lengthscale

Uses the same greedy information gain objective as the exact planner,
but with a Gibbs non-stationary kernel whose lengthscale l(x) is learned
online from data via marginal likelihood optimization.

Key difference from exact_planner.py:
- GP kernel has spatially varying lengthscale l(x)
- l(x) is parameterized by RBF basis functions (25 params on 5x5 grid)
- l(x) is optimized every N samples via GP marginal likelihood
- The non-stationary variance naturally concentrates samples near
  complex regions (short lengthscale = high info gain per sample)

Acquisition function (same as exact):
    x*_{k+1} = argmax  Delta(x_{k+1} | S_k) - lambda * c(x_k, x_{k+1})

But Delta(x) now uses the Gibbs kernel GP posterior variance, which
varies non-uniformly across the domain.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Float32, Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import torch
import time
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull

# Add nonstationary_planning module to path
import sys
script_dir = Path(__file__).parent
install_path = script_dir.parent / 'python3' / 'dist-packages'
if install_path.exists():
    sys.path.insert(0, str(install_path))
else:
    sys.path.insert(0, str(script_dir.parent))

from nonstationary_planning.gibbs_gp_model import NonstationaryGPModel

# Add info_gain package path for peak_detection (separate ROS2 package)
# Try installed path first (container: install/info_gain/lib/python3/dist-packages/)
# Then source path (host: src/info_gain/)
_ws_root = Path(__file__).resolve().parent
for _ in range(6):  # Walk up to find workspace root
    if (_ws_root / 'src' / 'info_gain' / 'info_gain').exists():
        sys.path.insert(0, str(_ws_root / 'src' / 'info_gain'))
        break
    if (_ws_root / 'install' / 'info_gain' / 'lib' / 'python3' / 'dist-packages').exists():
        sys.path.insert(0, str(_ws_root / 'install' / 'info_gain' / 'lib' / 'python3' / 'dist-packages'))
        break
    _ws_root = _ws_root.parent
from info_gain.peak_detection import detect_and_plot_peaks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_ground_truth_field(field_type, width=25.0, height=25.0, resolution=0.5):
    """Generate ground truth temperature field (same as field generators)"""
    x_grid = np.arange(0, width + 1e-9, resolution)
    y_grid = np.arange(0, height + 1e-9, resolution)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

    center_x, center_y = width / 2, height / 2
    base_temperature = 20.0
    hotspot_amplitude = 10.0

    if field_type == 'radial':
        sigma = 5.0
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'x_compress':
        sigma_x, sigma_y = 2.5, 7.0
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'y_compress':
        sigma_x, sigma_y = 7.0, 2.5
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'x_compress_tilt':
        sigma_x, sigma_y = 2.5, 7.0
        angle = np.pi / 4
        X_rot = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
        Y_rot = -(X - center_x) * np.sin(angle) + (Y - center_y) * np.cos(angle)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'y_compress_tilt':
        sigma_x, sigma_y = 7.0, 2.5
        angle = np.pi / 4
        X_rot = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
        Y_rot = -(X - center_x) * np.sin(angle) + (Y - center_y) * np.cos(angle)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return X, Y, field


def information_gain(variance, noise_var):
    """Delta(x) = (1/2) * log(1 + sigma^2(x) / sigma_n^2)"""
    if isinstance(variance, torch.Tensor):
        return 0.5 * torch.log(1 + variance / noise_var)
    else:
        return 0.5 * np.log(1 + variance / noise_var)


def travel_cost(x1, x2):
    """c(x1, x2) = ||x1 - x2||_2"""
    if isinstance(x1, torch.Tensor):
        return torch.norm(x1 - x2)
    else:
        return np.linalg.norm(np.array(x1) - np.array(x2))


class LiveVisualizer:
    """Visualization with lengthscale field panel"""

    def __init__(self, title="NS-Exact Sampler", output_dir=None, bounds=(0.0, 25.0, 0.0, 25.0), max_samples=100):
        self.fig = plt.figure(figsize=(20, 10))
        self.title = title
        self.output_dir = output_dir
        self.bounds = bounds
        self.max_samples = max_samples

        gs = gridspec.GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        self.ax_mean = self.fig.add_subplot(gs[0, 0])
        self.ax_var = self.fig.add_subplot(gs[0, 1])
        self.ax_acq = self.fig.add_subplot(gs[0, 2])
        self.ax_ls = self.fig.add_subplot(gs[0, 3])  # Lengthscale field
        self.ax_traj = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_cost = self.fig.add_subplot(gs[1, 2])
        self.ax_ls_hist = self.fig.add_subplot(gs[1, 3])  # Lengthscale history

        self.trajectory = []
        self.info_gains = []
        self.travel_costs = []
        self.steps = []
        self.l1_means = []
        self.l2_means = []
        self.aniso_means = []

        self.grid_res = 0.5
        x_lo, x_hi, y_lo, y_hi = bounds
        x = np.arange(x_lo, x_hi + self.grid_res, self.grid_res)
        y = np.arange(y_lo, y_hi + self.grid_res, self.grid_res)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.grid_points = np.column_stack([self.X_grid.ravel(), self.Y_grid.ravel()])

    def update(self, gp, candidates, scores, selected_idx, current_pos, target_pos,
               step, info_gain_val, cumulative_cost):
        self.trajectory.append(current_pos.copy())
        if info_gain_val > 0:
            self.info_gains.append(info_gain_val)
            self.travel_costs.append(cumulative_cost)
            self.steps.append(step)

        with torch.no_grad():
            grid_t = torch.tensor(self.grid_points, dtype=torch.float32).to(device)
            mean, var = gp.predict(grid_t)
            mean_grid = mean.cpu().numpy().reshape(self.X_grid.shape)
            var_grid = var.cpu().numpy().reshape(self.X_grid.shape)

        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

        # Get anisotropic lengthscale fields
        ls_X, ls_Y, ls_L1, ls_L2, ls_Theta = gp.get_lengthscale_field()
        self.l1_means.append(float(np.mean(ls_L1)))
        self.l2_means.append(float(np.mean(ls_L2)))
        self.aniso_means.append(float(np.mean(ls_L1 / ls_L2)))

        for ax in [self.ax_mean, self.ax_var, self.ax_acq, self.ax_ls,
                    self.ax_traj, self.ax_info, self.ax_cost, self.ax_ls_hist]:
            ax.clear()

        # 1. GP Mean
        self.ax_mean.pcolormesh(self.X_grid, self.Y_grid, mean_grid, cmap='coolwarm', shading='auto')
        self.ax_mean.set_title('GP Mean')
        self.ax_mean.set_aspect('equal')
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_mean.plot(traj[:, 0], traj[:, 1], 'k.-', linewidth=1, markersize=4)
        self.ax_mean.scatter(current_pos[0], current_pos[1], c='lime', s=100, marker='o', edgecolors='black', zorder=10)

        # 2. GP Variance
        self.ax_var.pcolormesh(self.X_grid, self.Y_grid, var_grid, cmap='viridis', shading='auto')
        self.ax_var.set_title('GP Variance')
        self.ax_var.set_aspect('equal')

        # 3. Acquisition Function
        self.ax_acq.scatter(candidates[:, 0], candidates[:, 1], c=scores_np, cmap='hot', s=30, alpha=0.7)
        if selected_idx is not None:
            sel = candidates[selected_idx]
            self.ax_acq.scatter(sel[0], sel[1], c='cyan', s=200, marker='X', edgecolors='black', linewidths=2, zorder=10)
        self.ax_acq.set_title(f'Acquisition (Step {step})')
        self.ax_acq.set_xlim(self.bounds[0], self.bounds[1])
        self.ax_acq.set_ylim(self.bounds[2], self.bounds[3])
        self.ax_acq.set_aspect('equal')

        # 4. Lengthscale field
        im_ls = self.ax_ls.pcolormesh(ls_X, ls_Y, ls_L1, cmap='plasma', shading='auto')
        self.ax_ls.set_title(f'l1(x) [{ls_L1.min():.2f}-{ls_L1.max():.2f}]')
        self.ax_ls.set_aspect('equal')
        plt.colorbar(im_ls, ax=self.ax_ls, label='l1(x) [m]')

        # 5. Trajectory
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_traj.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
            for i, pt in enumerate(traj):
                self.ax_traj.scatter(pt[0], pt[1], c='blue', s=50, zorder=5)
        self.ax_traj.set_title('Trajectory')
        self.ax_traj.set_xlim(self.bounds[0], self.bounds[1])
        self.ax_traj.set_ylim(self.bounds[2], self.bounds[3])
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True, alpha=0.3)

        # 6. Info gain
        if len(self.info_gains) > 0:
            self.ax_info.plot(self.steps, self.info_gains, 'g-o', linewidth=2, markersize=4)
        self.ax_info.set_title('Info Gain per Step')
        self.ax_info.set_xlabel('Step')
        self.ax_info.grid(True, alpha=0.3)

        # 7. Travel cost
        if len(self.travel_costs) > 0:
            self.ax_cost.plot(self.steps, self.travel_costs, 'r-o', linewidth=2, markersize=4)
        self.ax_cost.set_title('Cumulative Travel')
        self.ax_cost.set_xlabel('Step')
        self.ax_cost.grid(True, alpha=0.3)

        # 8. Lengthscale evolution
        ls_steps = list(range(1, len(self.l1_means) + 1))
        self.ax_ls_hist.plot(ls_steps, self.l1_means, 'purple', linewidth=2, label='l1')
        self.ax_ls_hist.plot(ls_steps, self.l2_means, 'orange', linewidth=2, label='l2')
        self.ax_ls_hist.plot(ls_steps, self.aniso_means, 'green', linewidth=1, linestyle='--', label='aniso')
        self.ax_ls_hist.set_title('Lengthscale Evolution')
        self.ax_ls_hist.set_xlabel('Step')
        self.ax_ls_hist.set_ylabel('l(x) [m]')
        self.ax_ls_hist.legend(fontsize=8)
        self.ax_ls_hist.grid(True, alpha=0.3)

        self.fig.suptitle(f'{self.title} - Sample {step}/{self.max_samples}', fontsize=14, fontweight='bold')

        if self.output_dir:
            progress_path = self.output_dir / 'figures' / 'progress.png'
            self.fig.savefig(progress_path, dpi=100, bbox_inches='tight')

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)


class NonstationaryExactSampler(Node):
    """Non-stationary exact planner with Gibbs kernel"""

    def __init__(self):
        super().__init__('nonstationary_exact_sampler')

        # ROS2 Parameters
        self.declare_parameter('field_type', 'radial')
        self.declare_parameter('trial', -1)
        self.declare_parameter('noise_var', 0.36)
        self.declare_parameter('lengthscale', 2.0)
        self.declare_parameter('lambda_cost', 0.1)
        self.declare_parameter('candidate_resolution', 1.0)
        self.declare_parameter('optimize_every', 10)
        self.declare_parameter('optimize_steps', 50)
        self.declare_parameter('grid_size', 5)
        self.declare_parameter('l_min', 0.5)
        self.declare_parameter('output_root', '/home/blazair/workspaces/boat_adaptive/data')
        self.declare_parameter('l_max', 5.0)

        # Sampling-box parameters (boat port)
        self.declare_parameter('field_origin_x', 0.0)
        self.declare_parameter('field_origin_y', 0.0)
        self.declare_parameter('field_size_x', 25.0)
        self.declare_parameter('field_size_y', 25.0)
        self.declare_parameter('candidate_edge_buffer', 0.5)
        self.declare_parameter('waypoint_tolerance', 0.5)
        self.declare_parameter('max_samples', 100)
        self.declare_parameter('temp_topic', '')
        self.declare_parameter('initial_waypoint_fractions', [0.2, 0.2, 0.4, 0.2, 0.4, 0.4])
        self.declare_parameter('evaluation_enabled', True)
        self.declare_parameter('publish_recon', True)
        self.declare_parameter('recon_temp_min', 10.0)
        self.declare_parameter('recon_temp_max', 35.0)
        self.declare_parameter('wait_for_preflight', False)
        self.declare_parameter('preflight_timeout', 60.0)

        self.field_type = self.get_parameter('field_type').value
        self.trial_num = self.get_parameter('trial').value
        self.noise_var = self.get_parameter('noise_var').value
        self.lengthscale = self.get_parameter('lengthscale').value
        self.lambda_cost = self.get_parameter('lambda_cost').value
        self.candidate_res = self.get_parameter('candidate_resolution').value
        self.optimize_every = self.get_parameter('optimize_every').value
        self.optimize_steps = self.get_parameter('optimize_steps').value
        self.grid_size = self.get_parameter('grid_size').value
        self.l_min = self.get_parameter('l_min').value
        self.l_max = self.get_parameter('l_max').value

        self.field_origin_x = float(self.get_parameter('field_origin_x').value)
        self.field_origin_y = float(self.get_parameter('field_origin_y').value)
        self.field_size_x = float(self.get_parameter('field_size_x').value)
        self.field_size_y = float(self.get_parameter('field_size_y').value)
        self.candidate_edge_buffer = float(self.get_parameter('candidate_edge_buffer').value)
        self.waypoint_tolerance = float(self.get_parameter('waypoint_tolerance').value)
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.evaluation_enabled = bool(self.get_parameter('evaluation_enabled').value)
        self.publish_recon = bool(self.get_parameter('publish_recon').value)
        self.recon_temp_min = float(self.get_parameter('recon_temp_min').value)
        self.recon_temp_max = float(self.get_parameter('recon_temp_max').value)
        self.wait_for_preflight = bool(self.get_parameter('wait_for_preflight').value)
        self.preflight_timeout = float(self.get_parameter('preflight_timeout').value)

        temp_topic_param = self.get_parameter('temp_topic').value
        self.temp_topic = temp_topic_param if temp_topic_param else f'/gaussian_field/{self.field_type}/temperature_noisy'

        # QoS profile for PX4
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # MAVROS setpoint topic uses default reliable QoS
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.info_pub = self.create_publisher(Float32, '/info_gain/current', 10)
        self.cost_pub = self.create_publisher(Float32, '/info_gain/cumulative_cost', 10)
        self.variance_pub = self.create_publisher(Float32, '/info_gain/mean_variance', 10)

        # Subscribers: ENU pose + pose-with-covariance from MAVROS
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos_sensor)
        self.pose_cov_sub = self.create_subscription(PoseWithCovarianceStamped, '/mavros/local_position/pose_cov', self.pose_cov_callback, qos_sensor)
        self.temp_sub = self.create_subscription(Float32, self.temp_topic, self.temp_callback, 10)

        # Live GP reconstruction marker (RViz) — latched
        if self.publish_recon:
            qos_latched = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.recon_pub = self.create_publisher(MarkerArray, '/info_gain/reconstruction', qos_latched)
        else:
            self.recon_pub = None

        # Preflight gate (latched).
        self.preflight_passed = not self.wait_for_preflight
        self._preflight_start_time = time.time()
        if self.wait_for_preflight:
            qos_latched_sub = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.preflight_sub = self.create_subscription(
                Bool, '/preflight/passed', self._preflight_callback, qos_latched_sub)

        # MAVROS service clients for arming + mode switching
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # State
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.position_variance = np.zeros(2)
        self.current_temp = None
        self.counter = 0
        self.state = 'WAIT_PREFLIGHT' if self.wait_for_preflight else 'INIT'
        self.current_target = np.array([0.0, 0.0, 0.0])
        self.sample_count = 0
        self.waiting_for_observation = False
        self.last_command_time = None
        self.total_travel_cost = 0.0
        self.last_position = None
        self.cumulative_info_gain = 0.0
        self.stop_reason = None

        # v1.2: stagnation detector deleted (see exact_planner.py).

        # Initial waypoints — built from launch-arg fractions of the configured box.
        fractions = list(self.get_parameter('initial_waypoint_fractions').value)
        if len(fractions) < 2 or len(fractions) % 2 != 0:
            self.get_logger().warn(
                f'initial_waypoint_fractions has odd length {len(fractions)}; using default')
            fractions = [0.2, 0.2, 0.4, 0.2, 0.4, 0.4]
        self.initial_waypoints = []
        for i in range(0, len(fractions), 2):
            fx, fy = float(fractions[i]), float(fractions[i + 1])
            wx = self.field_origin_x + fx * self.field_size_x
            wy = self.field_origin_y + fy * self.field_size_y
            self.initial_waypoints.append(np.array([wx, wy, 0.0]))
        self.waypoint_idx = 0

        # Non-stationary GP model (Gibbs kernel)
        self.gp = NonstationaryGPModel(
            noise_var=self.noise_var,
            signal_var=1.0,
            l_init=self.lengthscale,
            grid_size=self.grid_size,
            l_min=self.l_min,
            l_max=self.l_max,
            optimize_every=self.optimize_every,
            optimize_steps=self.optimize_steps,
        )

        # CUDA warmup
        self.get_logger().info('Warming up CUDA/PyTorch with Gibbs kernel...')
        dummy_X = np.array([[12.5, 12.5], [10.0, 10.0], [15.0, 15.0]])
        dummy_y = np.array([25.0, 24.0, 23.0])
        self.gp.fit(dummy_X, dummy_y)
        test_pts = torch.tensor([[12.0, 12.0], [13.0, 13.0]], dtype=torch.float32)
        _ = self.gp.predict(test_pts)
        self.gp = NonstationaryGPModel(
            noise_var=self.noise_var, signal_var=1.0, l_init=self.lengthscale,
            grid_size=self.grid_size, l_min=self.l_min, l_max=self.l_max,
            optimize_every=self.optimize_every, optimize_steps=self.optimize_steps,
        )
        self.get_logger().info('CUDA warmup complete')

        # Candidate grid — uses configured origin/size and edge buffer
        buf = self.candidate_edge_buffer
        self.candidates = self._generate_candidate_grid(
            self.field_origin_x + buf,
            self.field_origin_x + self.field_size_x - buf,
            self.field_origin_y + buf,
            self.field_origin_y + self.field_size_y - buf,
            self.candidate_res,
        )

        # Output directory
        self.output_dir = self._create_trial_directory()

        # Ground truth (sim only)
        if self.evaluation_enabled:
            self.get_logger().info('Generating ground truth field...')
            try:
                self.gt_X, self.gt_Y, self.gt_field = generate_ground_truth_field(self.field_type)
                self._save_ground_truth()
            except ValueError as e:
                self.get_logger().warn(f'Ground truth unavailable ({e}); disabling evaluation')
                self.evaluation_enabled = False
                self.gt_X = self.gt_Y = self.gt_field = None
        else:
            self.get_logger().info('evaluation_enabled=False; skipping ground truth generation')
            self.gt_X = self.gt_Y = self.gt_field = None

        # Data logging
        self.samples = []
        self.decisions = []
        self.samples_file = self.output_dir / 'samples.csv'
        self._init_samples_csv()

        # P3: per-trial spectral-fit log.  GP model appends a row each
        # time update_kernel_from_data is called (accepted or rejected).
        self.gp.set_fit_log_path(
            self.output_dir / 'fit_log.csv',
            step_fn=lambda: len(self.samples))

        # Visualization
        self.get_logger().info('Creating live visualizer...')
        try:
            self.viz = LiveVisualizer(
                title=f'NS-Exact - {self.field_type}',
                output_dir=self.output_dir,
                bounds=(self.field_origin_x, self.field_origin_x + self.field_size_x,
                        self.field_origin_y, self.field_origin_y + self.field_size_y),
                max_samples=self.max_samples,
            )
        except Exception as e:
            self.get_logger().error(f'Failed to create visualizer: {e}')
            self.viz = None

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self._save_config()

        self.get_logger().info(f'='*60)
        self.get_logger().info(f'Non-Stationary EXACT Sampler initialized')
        self.get_logger().info(f'  Kernel: Gibbs (spatially varying lengthscale)')
        self.get_logger().info(f'  Grid size: {self.grid_size}x{self.grid_size} ({self.grid_size**2} basis)')
        self.get_logger().info(f'  l range: [{self.l_min}, {self.l_max}], init={self.lengthscale}')
        self.get_logger().info(f'  Optimize every: {self.optimize_every} samples')
        self.get_logger().info(f'  Field: {self.field_type}')
        self.get_logger().info(f'  Trial: {self.trial_num}')
        self.get_logger().info(f'  Box: origin=({self.field_origin_x:.1f},{self.field_origin_y:.1f}) '
                               f'size=({self.field_size_x:.1f}x{self.field_size_y:.1f}) m')
        self.get_logger().info(f'  Waypoint tol: {self.waypoint_tolerance:.2f} m')
        self.get_logger().info(f'  Temp topic: {self.temp_topic}')
        self.get_logger().info(f'  Evaluation: {self.evaluation_enabled}')
        self.get_logger().info(f'  Wait for preflight: {self.wait_for_preflight}')
        self.get_logger().info(f'  Output: {self.output_dir}')
        self.get_logger().info(f'='*60)

    def _preflight_callback(self, msg):
        if msg.data and not self.preflight_passed:
            self.preflight_passed = True
            self.get_logger().info('Preflight passed — releasing planner from WAIT_PREFLIGHT')

    def _generate_candidate_grid(self, x_min, x_max, y_min, y_max, resolution):
        x = np.arange(x_min, x_max + 1e-9, resolution)
        y = np.arange(y_min, y_max + 1e-9, resolution)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _create_trial_directory(self):
        # Default lives inside the boat_adaptive workspace; override via
        # the `output_root` ROS2 parameter if you want it elsewhere.
        output_root = Path(self.get_parameter('output_root').value)
        base_dir = output_root / 'trials' / 'nonstationary_exact' / self.field_type
        base_dir.mkdir(parents=True, exist_ok=True)

        if self.trial_num >= 0:
            trial_num = self.trial_num
        else:
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
            trial_num = max([int(d.name.split('_')[1]) for d in existing], default=0) + 1

        trial_dir = base_dir / f'trial_{trial_num:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / 'figures').mkdir(exist_ok=True)
        (trial_dir / 'lengthscale').mkdir(exist_ok=True)
        self.trial_num = trial_num
        return trial_dir

    def _save_config(self):
        config = {
            'method': 'nonstationary_exact',
            'description': 'Non-stationary Gibbs kernel GP, exact position, greedy info gain',
            'kernel': 'gibbs',
            'field_type': self.field_type,
            'trial': self.trial_num,
            'noise_var': self.noise_var,
            'lengthscale_init': self.lengthscale,
            'lambda_cost': self.lambda_cost,
            'candidate_resolution': self.candidate_res,
            'grid_size': self.grid_size,
            'l_min': self.l_min,
            'l_max': self.l_max,
            'optimize_every': self.optimize_every,
            'optimize_steps': self.optimize_steps,
            'max_samples': self.max_samples,
            'field_origin': [self.field_origin_x, self.field_origin_y],
            'field_size': [self.field_size_x, self.field_size_y],
            'waypoint_tolerance': self.waypoint_tolerance,
            'evaluation_enabled': self.evaluation_enabled,
            'n_initial': len(self.initial_waypoints),
            'n_candidates': len(self.candidates),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def _save_ground_truth(self):
        if not self.evaluation_enabled or self.gt_field is None:
            return
        gt_file = self.output_dir / 'ground_truth.npz'
        np.savez(gt_file, X=self.gt_X, Y=self.gt_Y, field=self.gt_field)

    def _init_samples_csv(self):
        self._csv_fieldnames = ['step', 'phase', 'x', 'y', 'temp', 'info_gain', 'cumulative_info',
                                'travel_cost', 'gp_n_obs', 'pos_var_x', 'pos_var_y', 'pos_std_x', 'pos_std_y',
                                'ls_optimized', 'l1_mean', 'l2_mean', 'theta_mean', 'aniso_ratio']
        with open(self.samples_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()

    def _write_sample(self, sample_dict):
        try:
            with open(self.samples_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writerow(sample_dict)
        except Exception as e:
            self.get_logger().warn(f'Failed to write sample to CSV: {e}')

    def _maybe_optimize_lengthscale(self):
        """Optimize Gibbs kernel lengthscale if it's time."""
        if self.gp.should_optimize():
            self.get_logger().info(f'Optimizing lengthscale (n={self.gp.n_observations})...')
            t0 = time.time()
            self.gp.optimize_lengthscale(logger=self.get_logger())
            dt = time.time() - t0
            self.get_logger().info(f'Lengthscale optimization took {dt:.2f}s')

            # Save snapshot
            self.gp.save_lengthscale_snapshot(
                self.output_dir / 'lengthscale',
                self.sample_count
            )
            return True
        return False

    def _get_ls_stats(self):
        """Get current anisotropic lengthscale statistics."""
        _, _, L1, L2, Theta = self.gp.get_lengthscale_field()
        return (float(np.mean(L1)), float(np.mean(L2)),
                float(np.mean(Theta)), float(np.mean(L1 / L2)))

    def _compute_reconstruction_metrics(self):
        grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
        grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

        with torch.no_grad():
            gp_mean, gp_var = self.gp.predict(grid_t)
            gp_mean = gp_mean.cpu().numpy().reshape(self.gt_X.shape)
            gp_var = gp_var.cpu().numpy().reshape(self.gt_X.shape)

        error = gp_mean - self.gt_field
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        max_error = float(np.max(np.abs(error)))
        mean_variance = float(np.mean(gp_var))

        metrics = {
            'rmse': rmse, 'mae': mae, 'max_error': max_error,
            'mean_variance': mean_variance, 'n_observations': self.gp.n_observations
        }

        with open(self.output_dir / 'reconstruction_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        np.savez(self.output_dir / 'gp_reconstruction.npz',
                 X=self.gt_X, Y=self.gt_Y, mean=gp_mean, variance=gp_var, error=error)

        self._plot_reconstruction_comparison(gp_mean, gp_var, error)
        self.get_logger().info(f'Reconstruction: RMSE={rmse:.3f}, MAE={mae:.3f}, Max={max_error:.3f}')
        return metrics

    def _plot_reconstruction_comparison(self, gp_mean, gp_var, error):
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        samples_arr = None
        if len(self.samples) > 0:
            samples_arr = np.array([[s['x'], s['y']] for s in self.samples])

        # Ground truth
        im0 = axes[0, 0].pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto')
        axes[0, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, 0], label='T [°C]')
        if samples_arr is not None:
            axes[0, 0].scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=30, marker='x', linewidths=2)

        # GP Reconstruction
        im1 = axes[0, 1].pcolormesh(self.gt_X, self.gt_Y, gp_mean, cmap='coolwarm', shading='auto')
        axes[0, 1].set_title(f'GP Reconstruction (n={self.gp.n_observations})', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 1], label='T [°C]')

        # Error
        im2 = axes[0, 2].pcolormesh(self.gt_X, self.gt_Y, np.abs(error), cmap='hot', shading='auto')
        axes[0, 2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 2], label='|Error| [°C]')
        axes[0, 2].text(0.02, 0.98, f'RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C',
                       transform=axes[0, 2].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Variance
        im3 = axes[1, 0].pcolormesh(self.gt_X, self.gt_Y, gp_var, cmap='viridis', shading='auto')
        axes[1, 0].set_title(f'GP Variance', fontsize=12, fontweight='bold')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0], label='Variance')

        # Lengthscale field
        ls_X, ls_Y, ls_L1, ls_L2, ls_Theta = self.gp.get_lengthscale_field()
        im4 = axes[1, 1].pcolormesh(ls_X, ls_Y, ls_L1, cmap='plasma', shading='auto')
        axes[1, 1].set_title(f'Learned l1(x) [aniso={np.mean(ls_L1/ls_L2):.2f}]', fontsize=12, fontweight='bold')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 1], label='l1(x) [m]')
        if samples_arr is not None:
            axes[1, 1].scatter(samples_arr[:, 0], samples_arr[:, 1], c='white', s=20, edgecolors='black')

        # Convex hull
        if samples_arr is not None and len(samples_arr) >= 3:
            try:
                hull = ConvexHull(samples_arr)
                hull_area = hull.volume
                box_area = max(self.field_size_x * self.field_size_y, 1e-6)
                coverage = (hull_area / box_area) * 100

                axes[1, 2].pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto', alpha=0.6)
                axes[1, 2].scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=50, marker='o', edgecolors='white', linewidths=2)
                for simplex in hull.simplices:
                    axes[1, 2].plot(samples_arr[simplex, 0], samples_arr[simplex, 1], 'r-', linewidth=2)
                hull_points = samples_arr[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                axes[1, 2].fill(hull_points[:, 0], hull_points[:, 1], color='yellow', alpha=0.3)
                axes[1, 2].set_title(f'Coverage {coverage:.1f}%', fontsize=12, fontweight='bold')
            except Exception:
                axes[1, 2].set_title('Coverage', fontsize=12, fontweight='bold')
        x_lo, x_hi = self.field_origin_x, self.field_origin_x + self.field_size_x
        y_lo, y_hi = self.field_origin_y, self.field_origin_y + self.field_size_y
        axes[1, 2].set_xlim(x_lo, x_hi)
        axes[1, 2].set_ylim(y_lo, y_hi)
        axes[1, 2].set_aspect('equal')

        plt.suptitle(f'NS-Exact - {self.field_type} (Trial {self.trial_num}) - RMSE: {rmse:.3f}°C',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def pose_callback(self, msg):
        """Position from MAVROS /mavros/local_position/pose (ENU)."""
        p = msg.pose.position
        self.current_position = np.array([p.x, p.y, p.z])

    def pose_cov_callback(self, msg):
        """Position variance from /mavros/local_position/pose_cov diagonal."""
        cov = msg.pose.covariance
        self.position_variance = np.array([float(cov[0]), float(cov[7])])

    def temp_callback(self, msg):
        self.current_temp = msg.data

    def control_loop(self):
        # MAVROS requires a >2 Hz setpoint stream before OFFBOARD can hold
        self.publish_setpoint()

        if self.state == 'WAIT_PREFLIGHT':
            if self.preflight_passed:
                self.state = 'INIT'
                self.get_logger().info('Preflight cleared, entering INIT')
            elif time.time() - self._preflight_start_time > self.preflight_timeout:
                self.get_logger().error(
                    f'Preflight timeout after {self.preflight_timeout:.0f}s — aborting')
                self.state = 'DONE'
                self.stop_reason = 'preflight_timeout'
            return

        if self.state == 'INIT':
            self.counter += 1
            if self.counter >= 10:
                self.engage_offboard()
                self.arm()
                self.state = 'ARM'
                self.last_command_time = time.time()
                self._arm_retry_time = time.time()
                self.get_logger().info('Sent OFFBOARD + arm, waiting 3s...')

        elif self.state == 'ARM':
            if time.time() - self._arm_retry_time >= 0.5:
                self.engage_offboard()
                self.arm()
                self._arm_retry_time = time.time()
            if time.time() - self.last_command_time >= 3.0:
                self.state = 'INITIAL_SAMPLING'
                self.current_target = self.initial_waypoints[0]
                self.last_position = self.current_position[:2].copy()
                self._stuck_check_time = time.time()
                self.get_logger().info('Starting initial sampling (3 points)')

        elif self.state == 'INITIAL_SAMPLING':
            self._run_initial_sampling()

        elif self.state == 'ADAPTIVE_SAMPLING':
            self._run_adaptive_sampling()

    def _run_initial_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        origin_xy = np.array([self.field_origin_x, self.field_origin_y])
        if hasattr(self, '_stuck_check_time') and np.linalg.norm(self.current_position[:2] - origin_xy) < 1.0:
            if time.time() - self._stuck_check_time > 15.0:
                self.get_logger().warn('Vehicle stuck - re-sending arm + offboard')
                self.arm()
                self.engage_offboard()
                self._stuck_check_time = time.time()
        else:
            self._stuck_check_time = time.time()

        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        if time.time() - self._last_log_time > 5.0:
            self.get_logger().info(
                f'Initial sampling: wp {self.waypoint_idx+1}/{len(self.initial_waypoints)}, dist={dist:.2f}m'
            )
            self._last_log_time = time.time()

        if dist < self.waypoint_tolerance and self.current_temp is not None:
            x = self.current_position[:2].copy()
            y = self.current_temp

            if self.last_position is not None:
                self.total_travel_cost += travel_cost(self.last_position, x)
            self.last_position = x.copy()

            self.gp.add_observation(x, y)
            self.sample_count += 1

            l1_mean, l2_mean, theta_mean, aniso_ratio = self._get_ls_stats()
            sample = {
                'step': self.sample_count, 'phase': 'initial',
                'x': float(x[0]), 'y': float(x[1]), 'temp': float(y),
                'info_gain': 0.0, 'cumulative_info': 0.0,
                'travel_cost': float(self.total_travel_cost),
                'gp_n_obs': self.gp.n_observations,
                'pos_var_x': float(self.position_variance[0]),
                'pos_var_y': float(self.position_variance[1]),
                'pos_std_x': float(np.sqrt(self.position_variance[0])),
                'pos_std_y': float(np.sqrt(self.position_variance[1])),
                'ls_optimized': False, 'l1_mean': l1_mean, 'l2_mean': l2_mean,
                'theta_mean': theta_mean, 'aniso_ratio': aniso_ratio,
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.get_logger().info(f'Initial {self.waypoint_idx + 1}/3: ({x[0]:.1f}, {x[1]:.1f}), temp={y:.2f}')

            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.initial_waypoints):
                self.state = 'ADAPTIVE_SAMPLING'
                self.get_logger().info('='*60)
                self.get_logger().info('Starting adaptive sampling (non-stationary)')
                self.get_logger().info('='*60)
                self._plan_next_sample()
            else:
                self.current_target = self.initial_waypoints[self.waypoint_idx]

    def _run_adaptive_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        if self.waiting_for_observation and dist < self.waypoint_tolerance and self.current_temp is not None:
            x = self.current_position[:2].copy()
            y = self.current_temp

            # Info gain BEFORE updating GP
            _, var_at_x = self.gp.predict(torch.tensor(x.reshape(1, -1), dtype=torch.float32))
            realized_info = float(information_gain(var_at_x, self.noise_var).item())
            self.cumulative_info_gain += realized_info

            step_cost = travel_cost(self.last_position, x)
            self.total_travel_cost += step_cost
            self.last_position = x.copy()

            self.gp.add_observation(x, y)
            self.sample_count += 1

            # Optimize lengthscale if it's time
            optimized = self._maybe_optimize_lengthscale()

            l1_mean, l2_mean, theta_mean, aniso_ratio = self._get_ls_stats()
            sample = {
                'step': self.sample_count, 'phase': 'adaptive',
                'x': float(x[0]), 'y': float(x[1]), 'temp': float(y),
                'info_gain': realized_info,
                'cumulative_info': float(self.cumulative_info_gain),
                'travel_cost': float(self.total_travel_cost),
                'gp_n_obs': self.gp.n_observations,
                'pos_var_x': float(self.position_variance[0]),
                'pos_var_y': float(self.position_variance[1]),
                'pos_std_x': float(np.sqrt(self.position_variance[0])),
                'pos_std_y': float(np.sqrt(self.position_variance[1])),
                'ls_optimized': optimized, 'l1_mean': l1_mean, 'l2_mean': l2_mean,
                'theta_mean': theta_mean, 'aniso_ratio': aniso_ratio,
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.info_pub.publish(Float32(data=float(realized_info)))
            self.cost_pub.publish(Float32(data=float(self.total_travel_cost)))

            self.get_logger().info(
                f'Sample {self.sample_count}/{self.max_samples}: '
                f'({x[0]:.1f}, {x[1]:.1f}), info={realized_info:.4f}, '
                f'l1={l1_mean:.2f}, l2={l2_mean:.2f}, aniso={aniso_ratio:.2f}'
            )

            self.waiting_for_observation = False

            if self.sample_count >= self.max_samples:
                self.stop_reason = f'max_samples_reached ({self.max_samples})'
                self._finish_mission()
                return

            self._plan_next_sample()

    def _plan_next_sample(self):
        current_pos = self.current_position[:2].copy()
        best_idx, best_score, best_info, all_scores = self._greedy_single_step(current_pos)

        if best_idx is None:
            self.stop_reason = 'no_feasible_candidate'
            self._finish_mission()
            return

        x_next = self.candidates[best_idx]
        self.current_target = np.array([x_next[0], x_next[1], 0.0])
        self.waiting_for_observation = True

        scores_np = all_scores.cpu().numpy() if isinstance(all_scores, torch.Tensor) else all_scores
        top5_idx = np.argsort(scores_np)[-5:][::-1]
        top5_scores = scores_np[top5_idx]
        top5_pos = self.candidates[top5_idx]

        decision = {
            'step': self.sample_count + 1,
            'current_x': float(current_pos[0]), 'current_y': float(current_pos[1]),
            'selected_x': float(x_next[0]), 'selected_y': float(x_next[1]),
            'selected_score': float(best_score), 'selected_info': float(best_info),
            'travel_to_next': float(travel_cost(current_pos, x_next)),
            'lambda': self.lambda_cost,
            'n_candidates': len(self.candidates),
            'scores_min': float(scores_np.min()),
            'scores_max': float(scores_np.max()),
            'scores_mean': float(scores_np.mean()),
            'scores_std': float(scores_np.std()),
            'top5_x': top5_pos[:, 0].tolist(),
            'top5_y': top5_pos[:, 1].tolist(),
            'top5_scores': top5_scores.tolist(),
            'gp_n_obs': self.gp.n_observations,
        }
        self.decisions.append(decision)

        if self.viz is not None:
            try:
                self.viz.update(
                    gp=self.gp, candidates=self.candidates, scores=all_scores,
                    selected_idx=best_idx, current_pos=current_pos, target_pos=x_next,
                    step=self.sample_count + 1, info_gain_val=best_info,
                    cumulative_cost=self.total_travel_cost
                )
            except Exception as e:
                self.get_logger().warn(f'Viz update failed: {e}')

        if self.recon_pub is not None:
            try:
                self._publish_reconstruction()
            except Exception as e:
                self.get_logger().warn(f'Reconstruction publish failed: {e}')

        self.get_logger().info(f'Planned: ({x_next[0]:.1f}, {x_next[1]:.1f}), score={best_score:.4f}')

    def _publish_reconstruction(self):
        """Publish a CUBE_LIST MarkerArray of GP mean/variance on the candidate grid."""
        if self.recon_pub is None or self.gp.n_observations == 0:
            return

        candidates_t = torch.tensor(self.candidates, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, var = self.gp.predict(candidates_t)
            mean_np = mean.cpu().numpy()
            var_np = var.cpu().numpy()

        var_max = float(var_np.max()) if var_np.size else 1.0
        if var_max < 1e-9:
            var_max = 1.0

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gp_reconstruction'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.candidate_res
        marker.scale.y = self.candidate_res
        marker.scale.z = 0.05

        t_lo, t_hi = self.recon_temp_min, self.recon_temp_max
        t_range = max(t_hi - t_lo, 1e-6)

        for (cx, cy), m, v in zip(self.candidates, mean_np, var_np):
            pt = Point()
            pt.x = float(cx)
            pt.y = float(cy)
            pt.z = -0.05
            marker.points.append(pt)

            t = (float(m) - t_lo) / t_range
            t = max(0.0, min(1.0, t))
            col = ColorRGBA()
            col.r = float(t)
            col.g = float(0.3 + 0.4 * (1.0 - abs(2 * t - 1)))
            col.b = float(1.0 - t)
            alpha = 1.0 - float(v) / var_max
            col.a = max(0.2, min(1.0, alpha))
            marker.colors.append(col)

        ma = MarkerArray()
        ma.markers.append(marker)
        self.recon_pub.publish(ma)

    def _run_final_hotspot_analysis(self):
        """Run full Kac-Rice peak detection on the final GP posterior."""
        if self.evaluation_enabled and self.gt_X is not None:
            eval_X, eval_Y = self.gt_X, self.gt_Y
        else:
            res = 0.5
            x_lo, x_hi = self.field_origin_x, self.field_origin_x + self.field_size_x
            y_lo, y_hi = self.field_origin_y, self.field_origin_y + self.field_size_y
            xg = np.arange(x_lo, x_hi + 1e-9, res)
            yg = np.arange(y_lo, y_hi + 1e-9, res)
            eval_X, eval_Y = np.meshgrid(xg, yg, indexing='xy')

        try:
            grid_points = np.column_stack([eval_X.ravel(), eval_Y.ravel()])
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

            with torch.no_grad():
                mu, var = self.gp.predict(grid_t)
                mu_grid = mu.cpu().numpy().reshape(eval_X.shape)
                std_grid = np.sqrt(var.cpu().numpy().reshape(eval_X.shape))

            samples_arr = None
            if self.samples:
                samples_arr = np.array([[s['x'], s['y']] for s in self.samples])

            peaks = detect_and_plot_peaks(
                gp_wrapper=self.gp,
                lat_mesh=eval_X, lon_mesh=eval_Y,
                mu_orig=mu_grid, std_grid=std_grid,
                X_scaled=samples_arr,
                variable_name='Temperature',
                out_dir=self.output_dir,
                kernel_type='gibbs',
                y_mean=0.0,
                gp_label='NS-Exact GP',
            )

            n_sig = sum(1 for p in peaks if p.get('significant', False))
            self.get_logger().info(
                f'Final hotspot analysis: {len(peaks)} candidates, {n_sig} significant'
            )
        except Exception as e:
            self.get_logger().warn(f'Final hotspot analysis failed: {e}')

        if self.evaluation_enabled and self.gt_field is not None:
            self._run_ground_truth_hotspot_analysis()

    def _run_ground_truth_hotspot_analysis(self):
        """Fit GP to dense ground truth samples, run Kac-Rice for validation."""
        try:
            import shutil
            from info_gain.gp_model import GPModel as StationaryGPModel

            step = 3
            gt_sub_X = self.gt_X[::step, ::step]
            gt_sub_Y = self.gt_Y[::step, ::step]
            gt_sub_field = self.gt_field[::step, ::step]

            train_x = np.column_stack([gt_sub_X.ravel(), gt_sub_Y.ravel()])
            train_y = gt_sub_field.ravel()

            gt_gp = StationaryGPModel(noise_var=0.001, lengthscale=2.0)
            train_x_t = torch.tensor(train_x, dtype=torch.float32).to(device)
            train_y_t = torch.tensor(train_y, dtype=torch.float32).to(device)
            gt_gp.fit(train_x_t, train_y_t)

            grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, var = gt_gp.predict(grid_t)
                mu_grid = mu.cpu().numpy().reshape(self.gt_X.shape)
                std_grid = np.sqrt(var.cpu().numpy().reshape(self.gt_X.shape))

            gt_dir = self.output_dir / '_gt_hotspot_tmp'
            gt_dir.mkdir(exist_ok=True)

            peaks = detect_and_plot_peaks(
                gp_wrapper=gt_gp,
                lat_mesh=self.gt_X, lon_mesh=self.gt_Y,
                mu_orig=mu_grid, std_grid=std_grid,
                X_scaled=train_x,
                variable_name='Temperature (Ground Truth)',
                out_dir=gt_dir,
                kernel_type='rbf',
                y_mean=0.0,
                gp_label='Ground Truth GP',
            )

            for src_name, dst_name in [
                ('hotspots.png', 'ground_truth_hotspots.png'),
                ('peaks.csv', 'ground_truth_peaks.csv'),
            ]:
                src = gt_dir / src_name
                if src.exists():
                    shutil.move(str(src), str(self.output_dir / dst_name))

            shutil.rmtree(str(gt_dir), ignore_errors=True)

            n_sig = sum(1 for p in peaks if p.get('significant', False))
            self.get_logger().info(
                f'Ground truth hotspot analysis: {len(peaks)} candidates, {n_sig} significant'
            )
        except Exception as e:
            self.get_logger().warn(f'Ground truth hotspot analysis failed: {e}')

    def _greedy_single_step(self, current_pos):
        """Greedy single-step using Krause-Singh-Guestrin (2008) MI score.

        v1.2: replaced pointwise info-gain ½ log(1 + σ²/σ_n²) with
        the integrated variance-reduction acquisition.  Same change
        as in the stationary planners — no per-planner asymmetry.
        See info_gain/acquisition.py for math + paper citation.
        """
        from info_gain.acquisition import (
            mi_score, score_with_travel, info_gain_pointwise)

        candidates_np = np.asarray(self.candidates, dtype=float)
        mi = mi_score(self.gp, candidates_np, self.noise_var)
        scores_np = score_with_travel(
            mi, candidates_np, np.asarray(current_pos, dtype=float),
            self.lambda_cost)

        candidates_t = torch.tensor(self.candidates, dtype=torch.float32).to(device)
        _, variances = self.gp.predict(candidates_t)
        info_gains = info_gain_pointwise(variances, self.noise_var)

        scores = torch.tensor(scores_np, dtype=torch.float32).to(device)
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        best_info = float(info_gains[best_idx].item())

        return best_idx, best_score, best_info, scores

    def _finish_mission(self):
        self.state = 'DONE'

        # Save final lengthscale snapshot
        self.gp.save_lengthscale_snapshot(self.output_dir / 'lengthscale', self.sample_count)

        if self.evaluation_enabled and self.gt_field is not None:
            self.get_logger().info('Computing reconstruction metrics...')
            reconstruction_metrics = self._compute_reconstruction_metrics()
        else:
            self.get_logger().info('No ground truth — skipping reconstruction metrics')
            reconstruction_metrics = None

        # Run final Kac-Rice hotspot analysis (post-mission, on final GP)
        self.get_logger().info('Running final hotspot analysis...')
        self._run_final_hotspot_analysis()

        if self.viz is not None:
            self.viz.save(self.output_dir / 'figures' / 'final.png')

        with open(self.output_dir / 'samples.csv', 'w', newline='') as f:
            if self.samples:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.samples)

        # Save decisions CSV (detailed)
        with open(self.output_dir / 'decisions.csv', 'w', newline='') as f:
            if self.decisions:
                flat_decisions = []
                for d in self.decisions:
                    flat = {k: v for k, v in d.items() if not isinstance(v, list)}
                    flat['top5_x'] = str(d.get('top5_x', []))
                    flat['top5_y'] = str(d.get('top5_y', []))
                    flat['top5_scores'] = str(d.get('top5_scores', []))
                    flat_decisions.append(flat)
                writer = csv.DictWriter(f, fieldnames=flat_decisions[0].keys())
                writer.writeheader()
                writer.writerows(flat_decisions)

        with open(self.output_dir / 'decisions.json', 'w') as f:
            json.dump(self.decisions, f, indent=2)

        summary = {
            'method': 'nonstationary_exact',
            'kernel': 'gibbs',
            'field_type': self.field_type,
            'trial': self.trial_num,
            'lambda_cost': self.lambda_cost,
            'total_samples': self.sample_count,
            'total_travel_cost': float(self.total_travel_cost),
            'cumulative_info_gain': float(self.cumulative_info_gain),
            'stop_reason': self.stop_reason,
            'completed_at': datetime.now().isoformat()
        }
        if reconstruction_metrics is not None:
            summary.update({
                'reconstruction_rmse': reconstruction_metrics['rmse'],
                'reconstruction_mae': reconstruction_metrics['mae'],
                'reconstruction_max_error': reconstruction_metrics['max_error'],
                'mean_gp_variance': reconstruction_metrics['mean_variance'],
            })
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info('='*60)
        self.get_logger().info('MISSION COMPLETE (Non-Stationary Exact)')
        self.get_logger().info(f'  Samples: {self.sample_count}')
        self.get_logger().info(f'  Travel: {self.total_travel_cost:.1f}m')
        self.get_logger().info(f'  Info gain: {self.cumulative_info_gain:.4f}')
        if reconstruction_metrics is not None:
            self.get_logger().info(f'  RMSE: {reconstruction_metrics["rmse"]:.3f}°C')
        self.get_logger().info(f'  Data: {self.output_dir}')
        self.get_logger().info('='*60)

    def publish_setpoint(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.current_target[0])
        msg.pose.position.y = float(self.current_target[1])
        msg.pose.position.z = float(self.current_target[2])
        msg.pose.orientation.w = 1.0
        self.setpoint_pub.publish(msg)

    def arm(self):
        if not self.arm_client.service_is_ready():
            return
        req = CommandBool.Request()
        req.value = True
        self.arm_client.call_async(req)

    def engage_offboard(self):
        if not self.set_mode_client.service_is_ready():
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = 'OFFBOARD'
        self.set_mode_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = NonstationaryExactSampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        if node.viz is not None:
            node.viz.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
