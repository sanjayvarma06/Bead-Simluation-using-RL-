"""
RL-Tuned Pure Pursuit Gymnasium Environment
Implements the hybrid RL-PP environment according to:
'Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control with PPO'
(Elgouhary & El-Wakeel, 2026)
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from pathlib import Path

# Add directory and sibling directories to sys.path
_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_root_dir / "our_model_code") not in sys.path:
    sys.path.insert(0, str(_root_dir / "our_model_code"))

from environment import BeadEnvironment
from pure_pursuit import PurePursuitController, compute_path_curvature, compute_friction_speed_profile
from geometry_utils import points_to_polyline_distance


class RLPurePursuitEnv(gym.Env):
    """
    Gymnasium environment that trains an RL policy (PPO) to dynamically sequence
    lookahead distance L_d and steering gain g for a Pure Pursuit controller.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        image_path: Optional[str] = None,
        seed: int = 42,
        mode: str = "joint",  # "joint" (Ld, g), "ld_only" (Ld only, fixed g), "fixed", "adaptive"
        fixed_L: float = 0.55,
        fixed_g: float = 1.05,
    ):
        super().__init__()
        self.seed_val = seed
        np.random.seed(seed)
        self.mode = mode
        self.fixed_L = fixed_L
        self.fixed_g = fixed_g

        # Track list handling for diverse training across all paper datasets
        if isinstance(image_path, (list, tuple)):
            self.track_pool = list(image_path)
            init_img = self.track_pool[0]
        elif image_path in ("all_tracks", "tracks", "all"):
            self.track_pool = ["hockenheim_track.png", "montreal_track.png", "yasmarina_track.png", "path1.jpeg"]
            init_img = self.track_pool[0]
        else:
            self.track_pool = [image_path] if image_path else []
            init_img = image_path

        # Core arena environment
        self._env = BeadEnvironment(image_path=init_img, seed=seed)
        self.image_path = init_img

        # Pure Pursuit controller instance
        self.controller = PurePursuitController(
            wheelbase=0.33,
            L_min=0.20,
            L_max=1.20,
            g_min=0.60,
            g_max=1.40,
            beta_L=0.2,
            beta_g=0.2,
            max_steer=1.10,  # Curvature up to 6.0 m^-1 for sharp hairpins
        )

        # Precomputed/analyzed path curvature & speed profile (dynamically updated on reset)
        self.path_curvatures = np.empty(0, dtype=np.float32)
        self.speed_profile = np.empty(0, dtype=np.float32)
        self.is_closed = False

        # State tracking
        self.heading = 0.0
        self.speed = 0.0
        self.current_idx = 0
        self.prev_idx = 0
        self.step_count = 0
        self.max_steps = 2000
        self.lookahead_pt = np.zeros(2, dtype=np.float32)

        # Metrics
        self.covered_waypoints = set()
        self.strict_covered = set()
        self.trajectory = []

        # Observation space (Paper Eq. 1 + relative geometry):
        # [v/v_max, kappa0, kappa1, kappa2, delta_kappa, e_lat/R_cov, sin(e_theta), cos(e_theta)] -> 8 dimensions
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Action space (Paper Eq. 2):
        # Continuous 2D: [L_action, g_action] in [-1, 1]^2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._refresh_path_geometry()

    def _refresh_path_geometry(self):
        """Analyzes path curvature and speed profile dynamically for the current image."""
        pts = self._env.path_points
        N = len(pts)
        if N >= 3:
            # Check if contour is closed
            d_ends = float(np.linalg.norm(pts[0] - pts[-1]))
            self.is_closed = d_ends < 0.50 and len(self._env.path_breaks) == 0

            self.path_curvatures = compute_path_curvature(pts, is_closed=self.is_closed)
            self.speed_profile = compute_friction_speed_profile(
                self.path_curvatures,
                path_points=pts,
                is_closed=self.is_closed,
                v_base=4.2,
                v_min=1.8,
                v_max=5.0,
                alpha=2.5,
                a_brake=3.5,
                a_accel=2.5,
            )
        else:
            self.path_curvatures = np.zeros(max(1, N), dtype=np.float32)
            self.speed_profile = np.full(max(1, N), 3.5, dtype=np.float32)
            self.is_closed = False

    def reset(self, *, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)

        if len(self.track_pool) > 1:
            chosen_track = str(np.random.choice(self.track_pool))
            self.image_path = chosen_track
            self._env.load_contours(chosen_track)

        obs_core, _ = self._env.reset()
        self._refresh_path_geometry()

        pts = self._env.path_points
        if len(pts) > 1:
            # Initial heading tangential to first path segment
            dx = float(pts[1, 0] - pts[0, 0])
            dy = float(pts[1, 1] - pts[0, 1])
            self.heading = float(np.arctan2(dy, dx))
        else:
            self.heading = 0.0

        self.speed = 2.5
        self.current_idx = 0
        self.prev_idx = 0
        self.step_count = 0
        self.covered_waypoints.clear()
        self.strict_covered.clear()
        self.trajectory = [(float(self._env.pos_x), float(self._env.pos_y))]

        if len(pts) > 0:
            self.covered_waypoints.add(0)
            self.strict_covered.add(0)

        self.controller.reset(
            initial_L=self.fixed_L if self.mode in ("fixed", "ld_only") else (self.controller.L_min + self.controller.L_max) / 2.0,
            initial_g=self.fixed_g
        )

        obs = self._get_observation()
        info = self._get_info(0.0, 0.0)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        pos = np.array([self._env.pos_x, self._env.pos_y], dtype=np.float32)

        # 1. Decode Action depending on mode
        if self.mode == "joint":
            # Map [-1, 1] -> [L_min, L_max] and [g_min, g_max]
            a_L = float(np.clip(action[0], -1.0, 1.0))
            a_g = float(np.clip(action[1], -1.0, 1.0))
            raw_L = self.controller.L_min + 0.5 * (a_L + 1.0) * (self.controller.L_max - self.controller.L_min)
            raw_g = self.controller.g_min + 0.5 * (a_g + 1.0) * (self.controller.g_max - self.controller.g_min)
            L_d, g = self.controller.update_parameters(raw_L, raw_g)
        elif self.mode == "ld_only":
            a_L = float(np.clip(action[0], -1.0, 1.0))
            raw_L = self.controller.L_min + 0.5 * (a_L + 1.0) * (self.controller.L_max - self.controller.L_min)
            L_d, g = self.controller.update_parameters(raw_L, self.fixed_g)
        elif self.mode == "adaptive":
            # Linear v -> L_d baseline from paper Table II
            raw_L = float(np.clip(0.35 + 0.25 * self.speed, self.controller.L_min, self.controller.L_max))
            L_d, g = self.controller.update_parameters(raw_L, self.fixed_g)
        else:  # fixed
            L_d, g = self.controller.update_parameters(self.fixed_L, self.fixed_g)

        # 2. Find Lookahead point & nearest point on path using forward local window search
        pts = self._env.path_points
        N = len(pts)
        if N > 0:
            if self.is_closed:
                # Forward local search window around current_idx to prevent jumping across closed loops
                window_candidates = [(self.current_idx + k) % N for k in range(-3, 22)]
                sub_pts = pts[window_candidates]
                dists = np.hypot(sub_pts[:, 0] - pos[0], sub_pts[:, 1] - pos[1])
                min_k = int(np.argmin(dists))
                nearest_idx = window_candidates[min_k]
            else:
                window_start = max(0, self.current_idx - 3)
                window_end = min(N, self.current_idx + 22)
                sub_pts = pts[window_start:window_end]
                dists = np.hypot(sub_pts[:, 0] - pos[0], sub_pts[:, 1] - pos[1])
                nearest_idx = window_start + int(np.argmin(dists))

            # Continuous orthogonal distance to nearest polyline segment
            p0 = pts[(nearest_idx - 1) % N] if self.is_closed else pts[max(0, nearest_idx - 1)]
            p1 = pts[nearest_idx]
            p2 = pts[(nearest_idx + 1) % N] if self.is_closed else pts[min(N - 1, nearest_idx + 1)]

            def _seg_dist(p, a, b):
                ba = b - a
                pa = p - a
                denom = float(np.dot(ba, ba))
                if denom < 1e-12:
                    return float(np.linalg.norm(pa))
                t = float(np.clip(np.dot(pa, ba) / denom, 0.0, 1.0))
                return float(np.linalg.norm(pa - t * ba))

            nearest_dist = min(_seg_dist(pos, p0, p1), _seg_dist(pos, p1, p2))
            self.current_idx = nearest_idx

            # Forward lookahead point along path
            self.lookahead_pt, target_wp = self.controller.find_lookahead_point(
                pos, pts, nearest_idx, L_d, is_closed=self.is_closed
            )
        else:
            nearest_idx = 0
            nearest_dist = 0.0
            self.lookahead_pt = pos.copy()
            target_wp = 0

        # 3. Pure Pursuit Steering Law
        delta = self.controller.compute_steering(
            pos, self.heading, self.lookahead_pt, L_d, g
        )

        # 4. Reference Speed & Longitudinal Control
        target_speed = float(self.speed_profile[nearest_idx]) if len(self.speed_profile) > nearest_idx else 3.5
        accel = float(np.clip(3.0 * (target_speed - self.speed), -self._env.ACCEL, self._env.ACCEL))

        # 5. Vehicle Motion Update (Kinematic Bicycle dynamics, Eq. 7-8 in paper)
        dt = self._env.DT
        self.speed = float(np.clip(self.speed + accel * dt, 0.5, self._env.MAX_SPEED))
        
        # Heading rate: dot(theta) = (v / L) * tan(delta)
        yaw_rate = (self.speed / self.controller.wheelbase) * float(np.tan(delta))
        self.heading = float((self.heading + yaw_rate * dt + np.pi) % (2.0 * np.pi) - np.pi)

        # Update position
        vx = self.speed * float(np.cos(self.heading))
        vy = self.speed * float(np.sin(self.heading))
        self._env.pos_x += vx * dt
        self._env.pos_y += vy * dt
        self._env.vel_x = vx
        self._env.vel_y = vy

        # Enforce arena boundary collision
        hit_boundary = False
        if abs(self._env.pos_x) > self._env.BOUNDARY:
            self._env.pos_x = float(np.clip(self._env.pos_x, -self._env.BOUNDARY, self._env.BOUNDARY))
            self.heading = float(np.pi - self.heading)
            hit_boundary = True
        if abs(self._env.pos_y) > self._env.BOUNDARY:
            self._env.pos_y = float(np.clip(self._env.pos_y, -self._env.BOUNDARY, self._env.BOUNDARY))
            self.heading = float(-self.heading)
            hit_boundary = True

        self.trajectory.append((float(self._env.pos_x), float(self._env.pos_y)))

        # Update coverage metrics
        if nearest_dist <= self._env.COV_RADIUS:
            self.covered_waypoints.add(nearest_idx)
            # Advance sequential waypoint tracker in core env
            if self.is_closed:
                if (nearest_idx - self._env.waypoint_idx) % N < N // 2:
                    self._env.waypoint_idx = max(self._env.waypoint_idx, nearest_idx)
            else:
                if nearest_idx > self._env.waypoint_idx:
                    self._env.waypoint_idx = nearest_idx
        if nearest_dist <= self._env.STRICT_RADIUS:
            self.strict_covered.add(nearest_idx)

        # Compute progress: newly passed waypoints
        if self.is_closed:
            advanced = (nearest_idx - self.prev_idx) % N
            if advanced > N // 2:
                advanced = 0
        else:
            advanced = max(0, nearest_idx - self.prev_idx)
        self.prev_idx = nearest_idx

        # 6. Reward Computation
        reward = self._compute_paper_reward(
            L_d=L_d,
            g=g,
            nearest_idx=nearest_idx,
            nearest_dist=nearest_dist,
            advanced=advanced,
            hit_boundary=hit_boundary
        )

        # Termination criteria: complete circuit
        coverage = len(self.covered_waypoints) / max(1, N)
        sequential_cov = self._env.waypoint_idx / max(1, N - 1)
        
        lap_completed = False
        if self.is_closed:
            # Vehicle has navigated almost all waypoints and crossed back to the start region
            if self.step_count > 150 and self._env.waypoint_idx >= N - 8 and nearest_idx <= 8:
                lap_completed = True
            elif self.step_count > 150 and coverage >= 0.98:
                lap_completed = True
        else:
            if self.step_count > 50 and (self._env.waypoint_idx >= N - 4 or coverage >= 0.98):
                lap_completed = True

        terminated = bool(lap_completed)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        if terminated:
            reward += self._env.R_COMPLETION
        elif truncated:
            if coverage >= 0.80:
                reward += self._env.R_PARTIAL80
            else:
                reward += self._env.R_INCOMPLETE

        obs = self._get_observation()
        info = self._get_info(coverage, nearest_dist)
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["lookahead_distance"] = L_d
        info["steering_gain"] = g
        info["steering_angle"] = delta

        return obs, float(reward), terminated, truncated, info

    def _compute_paper_reward(
        self,
        L_d: float,
        g: float,
        nearest_idx: int,
        nearest_dist: float,
        advanced: int,
        hit_boundary: bool
    ) -> float:
        """
        Enhanced Paper Reward Equation with High-Precision Corridor & Corner Adaptation:
        - Exponential Gaussian tracking corridor: rewards staying within sub-4cm of centerline
        - Quadratic off-track penalty for deviations > 8cm
        - Dynamic lookahead shrink incentive before and during sharp bends
        - Steering gain responsiveness incentive on high-curvature segments
        - Heading alignment reward with local tangent
        - Table I teacher regularization and jerk penalties
        """
        N = len(self.path_curvatures)
        if N > 0:
            k0 = float(self.path_curvatures[nearest_idx % N])
            k1 = float(self.path_curvatures[(nearest_idx + 5) % N])
            k2 = float(self.path_curvatures[(nearest_idx + 12) % N])
        else:
            k0 = k1 = k2 = 0.0

        kappa_max = max(abs(k0), abs(k1), abs(k2))
        local_curv = abs(k0)

        # Teacher target
        L_star, g_star = self.controller.get_teacher_targets(
            self.speed, kappa_max, v_min=1.0, v_max=self._env.MAX_SPEED
        )

        r = 0.0

        # 1. High-Precision Centerline Corridor Reward (sub-4cm precision)
        sigma_tol = 0.035
        r += 3.5 * float(np.exp(-0.5 * (nearest_dist / sigma_tol) ** 2))

        # 2. Quadratic Off-Corridor Penalty (steep penalty if bead deviates > 0.07m)
        if nearest_dist > 0.07:
            excess = (nearest_dist - 0.07) / 0.07
            r -= 5.0 * float(min(excess ** 2, 9.0))

        # 3. Curvature-Adaptive Lookahead Incentive (shrink L_d on bends)
        if kappa_max > 0.65:
            if L_d > 0.40:
                r -= 4.0 * (L_d - 0.40) * kappa_max  # Penalize large lookahead on sharp turns
            else:
                r += 2.2  # Reward tight lookahead for sharp apex tracking

        # 4. Teacher Guidance Regularization
        r -= 2.0 * abs(L_d - L_star)
        r -= 1.0 * abs(g - g_star)

        # 5. Parameter Jerk / Smoothness Penalty
        r -= 0.25 * abs(L_d - self.controller.prev_L_smooth)
        r -= 0.10 * abs(g - self.controller.prev_g_smooth)

        # 6. Heading Alignment Incentive
        pts = self._env.path_points
        if len(pts) > nearest_idx + 1:
            nxt_idx = (nearest_idx + 1) % len(pts) if self.is_closed else min(len(pts) - 1, nearest_idx + 1)
            tang = pts[nxt_idx] - pts[nearest_idx]
            t_len = float(np.linalg.norm(tang))
            if t_len > 1e-5:
                path_hdg = float(np.arctan2(tang[1], tang[0]))
                e_th = (self.heading - path_hdg + np.pi) % (2.0 * np.pi) - np.pi
                r += 0.8 * float(np.cos(e_th))

        # 7. Waypoint Progression & Speed Efficiency
        r += 1.6 * float(advanced)
        r += 1.0 * (self.speed / self._env.MAX_SPEED)

        # 8. Collision / Boundary Penalty
        if hit_boundary:
            r -= 10.0

        # 9. Stall Penalty
        if self.speed < 1.0:
            r -= 1.0

        return float(np.clip(r, -30.0, 100.0))

    def _get_observation(self) -> np.ndarray:
        """
        Generates observation vector st = [vt, k0, k1, k2, delta_k, e_lat, sin(e_theta), cos(e_theta)].
        Purely relative geometric features -> Zero-shot invariant to global coordinates.
        """
        N = len(self.path_curvatures)
        pts = self._env.path_points
        pos = np.array([self._env.pos_x, self._env.pos_y], dtype=np.float32)

        if N > 0:
            idx = self.current_idx % N
            k0 = float(self.path_curvatures[idx])
            k1 = float(self.path_curvatures[(idx + 5) % N])
            k2 = float(self.path_curvatures[(idx + 12) % N])
            delta_k = k1 - k0

            # Lateral error and heading error relative to nearest path tangent
            next_idx = (idx + 1) % N
            tangent = pts[next_idx] - pts[idx]
            tangent_len = float(np.linalg.norm(tangent))
            if tangent_len > 1e-6:
                tangent_unit = tangent / tangent_len
                path_heading = float(np.arctan2(tangent_unit[1], tangent_unit[0]))
                normal = np.array([-tangent_unit[1], tangent_unit[0]], dtype=np.float32)
                e_lat = float(np.dot(pos - pts[idx], normal))
            else:
                path_heading = self.heading
                e_lat = 0.0

            e_theta = float((self.heading - path_heading + np.pi) % (2.0 * np.pi) - np.pi)
        else:
            k0 = k1 = k2 = delta_k = e_lat = e_theta = 0.0

        v_norm = float(self.speed / self._env.MAX_SPEED)
        k0_norm = float(np.clip(abs(k0) / 3.0, 0.0, 1.0))
        k1_norm = float(np.clip(abs(k1) / 3.0, 0.0, 1.0))
        k2_norm = float(np.clip(abs(k2) / 3.0, 0.0, 1.0))
        delta_k_norm = float(np.clip(delta_k / 2.0, -1.0, 1.0))
        e_lat_norm = float(np.clip(e_lat / self._env.COV_RADIUS, -1.0, 1.0))
        sin_eth = float(np.sin(e_theta))
        cos_eth = float(np.cos(e_theta))

        return np.array([
            v_norm,
            k0_norm,
            k1_norm,
            k2_norm,
            delta_k_norm,
            e_lat_norm,
            sin_eth,
            cos_eth
        ], dtype=np.float32)

    def _get_info(self, coverage: float, tracking_error: float) -> Dict[str, Any]:
        N = max(1, len(self._env.path_points))
        strict_cov = len(self.strict_covered) / N
        seq_cov = self._env.waypoint_idx / max(1, N - 1)

        return {
            "coverage": float(coverage),
            "sequential": float(seq_cov),
            "strict_coverage": float(strict_cov),
            "tracking_error": float(tracking_error),
            "speed": float(self.speed),
            "heading": float(self.heading),
            "lookahead_distance": float(self.controller.L_smooth),
            "steering_gain": float(self.controller.g_smooth),
            "steps": int(self.step_count)
        }

    def render(self):
        pass

    def close(self):
        pass
