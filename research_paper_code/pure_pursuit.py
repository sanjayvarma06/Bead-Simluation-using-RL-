"""
Pure Pursuit & Curvature Module
Implements the core geometric path tracking controller and curvature analysis from:
'Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control with PPO'
(Elgouhary & El-Wakeel, 2026)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def compute_path_curvature(path: np.ndarray, is_closed: bool = False, smoothing_window: int = 5) -> np.ndarray:
    """
    Computes signed and absolute discrete curvature along a 2D polyline {p_i = (x_i, y_i)}.
    Curvature kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
    Uses central differences with optional Gaussian/moving-average smoothing to reduce noise.
    """
    N = len(path)
    if N < 3:
        return np.zeros(N, dtype=np.float32)

    # Optional gentle smoothing on raw points before derivative computation
    if smoothing_window > 1 and N > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        # Pad ends
        pad_mode = 'wrap' if is_closed else 'edge'
        pad_len = smoothing_window // 2
        px = np.pad(path[:, 0], pad_len, mode=pad_mode)
        py = np.pad(path[:, 1], pad_len, mode=pad_mode)
        smooth_x = np.convolve(px, kernel, mode='valid')[:N]
        smooth_y = np.convolve(py, kernel, mode='valid')[:N]
        pts = np.stack([smooth_x, smooth_y], axis=1)
    else:
        pts = path

    # First and second derivatives
    if is_closed:
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
    else:
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5 + 1e-8
    kappa = numerator / denominator

    # Clip extreme spikes
    kappa = np.clip(kappa, -5.0, 5.0)
    return kappa.astype(np.float32)


def compute_friction_speed_profile(
    curvatures: np.ndarray,
    path_points: Optional[np.ndarray] = None,
    is_closed: bool = False,
    v_base: float = 4.5,
    v_min: float = 1.8,
    v_max: float = 5.5,
    alpha: float = 2.5,
    a_brake: float = 3.5,
    a_accel: float = 2.5,
) -> np.ndarray:
    """
    Computes a preview-braked friction-limited raceline speed profile:
    1. Static curvature limit: v_curv(s) = clip(v_base / sqrt(1 + alpha * |kappa|), v_min, v_max)
    2. Backward pass (preview deceleration): v_i = min(v_i, sqrt(v_{i+1}^2 + 2 * a_brake * ds_i))
    3. Forward pass (acceleration limit): v_{i+1} = min(v_{i+1}, sqrt(v_i^2 + 2 * a_accel * ds_i))
    """
    abs_k = np.abs(curvatures)
    v_prof = np.clip(v_base / np.sqrt(1.0 + alpha * abs_k), v_min, v_max).astype(np.float64)

    N = len(curvatures)
    if path_points is not None and N > 2 and len(path_points) == N:
        # Step distances
        if is_closed:
            ds = np.hypot(np.roll(path_points[:, 0], -1) - path_points[:, 0],
                          np.roll(path_points[:, 1], -1) - path_points[:, 1])
        else:
            ds = np.zeros(N, dtype=np.float64)
            ds[:-1] = np.hypot(np.diff(path_points[:, 0]), np.diff(path_points[:, 1]))
            ds[-1] = ds[-2] if N > 1 else 0.1

        # Backward deceleration pass (braking preview ahead of corners)
        passes = 2 if is_closed else 1
        for _ in range(passes):
            for i in range(N - 1, -1, -1):
                nxt = (i + 1) % N if is_closed else min(i + 1, N - 1)
                v_prof[i] = min(v_prof[i], np.sqrt(v_prof[nxt]**2 + 2.0 * a_brake * ds[i]))

        # Forward acceleration pass
        for _ in range(passes):
            for i in range(N):
                prev = (i - 1) % N if is_closed else max(0, i - 1)
                v_prof[i] = min(v_prof[i], np.sqrt(v_prof[prev]**2 + 2.0 * a_accel * ds[prev]))

    return np.clip(v_prof, v_min, v_max).astype(np.float32)


class PurePursuitController:
    """
    Geometric Pure Pursuit Controller with dynamic (L_d, g) tuning and first-order smoothing.
    Paper:
      Lookahead L_d in [L_min, L_max]
      Steering gain g in [g_min, g_max]
      Smoothing: L~_{t+1} = beta_L * L_{t+1} + (1 - beta_L) * L~_t
                 g~_{t+1} = beta_g * g_{t+1} + (1 - beta_g) * g~_t  (beta = 0.2)
      Steering: gamma = g~ * 2 * y' / L~_d^2
    """
    def __init__(
        self,
        wheelbase: float = 0.33,
        L_min: float = 0.20,
        L_max: float = 1.20,
        g_min: float = 0.60,
        g_max: float = 1.40,
        beta_L: float = 0.2,
        beta_g: float = 0.2,
        max_steer: float = 1.10,  # radians (~63 degrees; allows curvature up to ~6.0 m^-1 for sharp hairpins)
    ):
        self.wheelbase = wheelbase
        self.L_min = L_min
        self.L_max = L_max
        self.g_min = g_min
        self.g_max = g_max
        self.beta_L = beta_L
        self.beta_g = beta_g
        self.max_steer = max_steer

        # Smoothed state variables
        self.L_smooth = (L_min + L_max) / 2.0
        self.g_smooth = 1.0
        self.prev_L_smooth = self.L_smooth
        self.prev_g_smooth = self.g_smooth

    def reset(self, initial_L: Optional[float] = None, initial_g: Optional[float] = None):
        self.L_smooth = initial_L if initial_L is not None else (self.L_min + self.L_max) / 2.0
        self.g_smooth = initial_g if initial_g is not None else 1.0
        self.prev_L_smooth = self.L_smooth
        self.prev_g_smooth = self.g_smooth

    def update_parameters(self, L_raw: float, g_raw: float) -> Tuple[float, float]:
        """
        Applies clipping and first-order exponential smoothing (Eq. 3 in paper).
        """
        self.prev_L_smooth = self.L_smooth
        self.prev_g_smooth = self.g_smooth

        L_clipped = float(np.clip(L_raw, self.L_min, self.L_max))
        g_clipped = float(np.clip(g_raw, self.g_min, self.g_max))

        self.L_smooth = float(self.beta_L * L_clipped + (1.0 - self.beta_L) * self.prev_L_smooth)
        self.g_smooth = float(self.beta_g * g_clipped + (1.0 - self.beta_g) * self.prev_g_smooth)

        return self.L_smooth, self.g_smooth

    def get_teacher_targets(
        self,
        speed: float,
        kappa_max: float,
        v_min: float = 1.0,
        v_max: float = 7.0
    ) -> Tuple[float, float]:
        """
        Linear safety teacher targets (Eq. 5 in paper):
        L*_t = clip(0.50 + 0.28 * v_t - 3.5 * kappa_max, L_min, L_max)
        g*_t = clip(m * v_t + b, g_min, g_max)
        scaled appropriately for 10x10 arena path tracking.
        """
        # Teacher lookahead: shortens before bends (Ld ~ 0.28-0.38), extends on straights (Ld ~ 0.65-0.85)
        L_star = 0.65 - 0.18 * kappa_max + 0.04 * speed
        L_star = float(np.clip(L_star, 0.26, 0.85))

        # Teacher steering gain: increases on tight corners for crisp response
        g_star = 0.96 + 0.14 * kappa_max
        g_star = float(np.clip(g_star, self.g_min, self.g_max))

        return L_star, g_star

    def find_lookahead_point(
        self,
        vehicle_pos: np.ndarray,
        path: np.ndarray,
        current_idx: int,
        lookahead_dist: float,
        is_closed: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Finds the lookahead point on the path at distance lookahead_dist ahead of current position.
        Searches forward from current_idx along the polyline.
        """
        N = len(path)
        if N == 0:
            return vehicle_pos.copy(), 0
        if N == 1:
            return path[0].copy(), 0

        # Cumulative forward distance from current_idx
        idx = current_idx
        search_limit = N if is_closed else (N - current_idx)
        
        best_pt = path[(current_idx + 1) % N if is_closed else min(current_idx + 1, N - 1)]
        best_idx = (current_idx + 1) % N if is_closed else min(current_idx + 1, N - 1)

        for step in range(1, search_limit):
            target_idx = (current_idx + step) % N if is_closed else (current_idx + step)
            pt = path[target_idx]
            d = float(np.linalg.norm(pt - vehicle_pos))
            if d >= lookahead_dist:
                # Interpolate between previous point and target point for precision
                prev_idx = (target_idx - 1) % N if is_closed else max(0, target_idx - 1)
                p_prev = path[prev_idx]
                v = pt - p_prev
                seg_len = float(np.linalg.norm(v))
                if seg_len > 1e-6:
                    t = float(np.clip((lookahead_dist - np.linalg.norm(p_prev - vehicle_pos)) / seg_len, 0.0, 1.0))
                    best_pt = p_prev + t * v
                else:
                    best_pt = pt
                best_idx = target_idx
                break
            best_pt = pt
            best_idx = target_idx

        return np.asarray(best_pt, dtype=np.float32), int(best_idx)

    def compute_steering(
        self,
        vehicle_pos: np.ndarray,
        vehicle_heading: float,
        lookahead_pt: np.ndarray,
        lookahead_dist: Optional[float] = None,
        gain: Optional[float] = None
    ) -> float:
        """
        Pure Pursuit geometric steering law:
        Transform lookahead point into vehicle body frame: (x', y')
        Curvature command: gamma = g * (2 * y' / L_d^2)
        Kinematic steering angle: delta = arctan(wheelbase * gamma)
        """
        Ld = lookahead_dist if lookahead_dist is not None else self.L_smooth
        g = gain if gain is not None else self.g_smooth
        Ld = max(1e-4, Ld)

        dx = float(lookahead_pt[0] - vehicle_pos[0])
        dy = float(lookahead_pt[1] - vehicle_pos[1])

        # Rotation into vehicle coordinate system (x forward, y left)
        cos_h = np.cos(vehicle_heading)
        sin_h = np.sin(vehicle_heading)
        x_body = cos_h * dx + sin_h * dy
        y_body = -sin_h * dx + cos_h * dy

        # Curvature command gamma
        gamma = float(g * (2.0 * y_body) / (Ld ** 2))

        # Kinematic steering angle delta = arctan(wheelbase * gamma)
        steer_target = np.arctan(self.wheelbase * gamma)
        delta = float(np.clip(steer_target, -self.max_steer, self.max_steer))
        return delta
