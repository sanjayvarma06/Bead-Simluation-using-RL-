"""
Visualization & Interpretability Suite
Replicates Figures 6, 7, and 8 of:
'Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control with PPO'
(Elgouhary & El-Wakeel, 2026)
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_root_dir / "our_model_code") not in sys.path:
    sys.path.insert(0, str(_root_dir / "our_model_code"))

from stable_baselines3 import PPO
from rl_pure_pursuit_env import RLPurePursuitEnv
from geometry_utils import points_to_polyline_distance


def run_and_record(controller_mode: str, image_path: str, model_path: Optional[str] = None):
    env = RLPurePursuitEnv(image_path=image_path, mode=controller_mode, seed=15000)
    model = None
    if controller_mode in ("joint", "ld_only") and model_path and Path(model_path).exists():
        model = PPO.load(model_path, device="cpu")

    obs, info = env.reset()
    done = False
    trunc = False

    history = {
        "x": [env._env.pos_x],
        "y": [env._env.pos_y],
        "speed": [env.speed],
        "lookahead": [env.controller.L_smooth],
        "gain": [env.controller.g_smooth],
        "steering": [0.0],
        "waypoint_idx": [env.current_idx],
        "tracking_error": [0.0],
    }

    while not (done or trunc):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2, dtype=np.float32)

        obs, r, done, trunc, info = env.step(action)
        history["x"].append(env._env.pos_x)
        history["y"].append(env._env.pos_y)
        history["speed"].append(env.speed)
        history["lookahead"].append(info["lookahead_distance"])
        history["gain"].append(info["steering_gain"])
        history["steering"].append(info["steering_angle"])
        history["waypoint_idx"].append(env.current_idx)
        history["tracking_error"].append(info["tracking_error"])

    path = np.asarray(env._env.path_points, dtype=np.float64)
    curvatures = np.asarray(env.path_curvatures, dtype=np.float64)
    env.close()

    return history, path, curvatures


def plot_trajectory_overlay(image_path: str, model_path: str, out_path: str = "complete_model_trace.png"):
    """Replicates Fig. 6 & Fig. 7: Multi-controller trajectory overlay against reference path."""
    print(f"Generating Multi-Controller Trajectory Comparison for {image_path}...")
    
    hist_joint, ref_path, _ = run_and_record("joint", image_path, model_path)
    hist_adapt, _, _ = run_and_record("adaptive", image_path, None)
    hist_fixed, _, _ = run_and_record("fixed", image_path, None)

    traj_joint = np.stack([hist_joint["x"], hist_joint["y"]], axis=1)
    d_ref = points_to_polyline_distance(ref_path, traj_joint)
    geom10 = float(np.mean(d_ref <= 0.10))
    d_traj = points_to_polyline_distance(traj_joint, ref_path)
    rmse = float(np.sqrt(np.mean(d_traj ** 2)))

    fig, ax = plt.subplots(figsize=(10, 9))

    # Reference Path
    ax.plot(ref_path[:, 0], ref_path[:, 1], color="#7f8c8d", linestyle="--", linewidth=2.8, label="Reference Path", zorder=2)
    ax.scatter(ref_path[0, 0], ref_path[0, 1], color="#27ae60", s=140, marker="o", label="Start Point", zorder=6)

    # Baselines
    ax.plot(hist_fixed["x"], hist_fixed["y"], color="#e74c3c", linewidth=1.8, linestyle=":", alpha=0.85, label="Fixed PP (Ld=0.85m)", zorder=3)
    ax.plot(hist_adapt["x"], hist_adapt["y"], color="#f39c12", linewidth=2.0, linestyle="-.", alpha=0.9, label="Adaptive PP (Linear v -> Ld)", zorder=4)

    # Proposed RL-PP Joint
    ax.plot(hist_joint["x"], hist_joint["y"], color="#2980b9", linewidth=2.6, label="RL-PP Joint (Ld, g) [Proposed]", zorder=5)

    # Covered points highlight
    covered = d_ref <= 0.10
    ax.scatter(ref_path[covered, 0], ref_path[covered, 1], color="#2ecc71", s=12, alpha=0.5, label="Covered within 0.10m", zorder=1)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("Arena X (meters)", fontsize=11)
    ax.set_ylabel("Arena Y (meters)", fontsize=11)
    ax.set_title(
        f"RL-Tuned Pure Pursuit vs Baselines (Elgouhary & El-Wakeel 2026)\n"
        f"Joint RL-PP Geometric@0.10: {geom10*100:.2f}% | RMSE: {rmse:.4f}m | Steps: {len(hist_joint['x'])-1}",
        fontsize=12, pad=12
    )
    ax.legend(loc="best", framealpha=0.92, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Trajectory Overlay Plot -> {out_path}")


def plot_interpretability(image_path: str, model_path: str, out_path: str = "rl_pp_interpretability.png"):
    """Replicates Fig. 8 of the paper: Along-lap evolution and schedule vs curvature/speed."""
    print(f"Generating Policy Interpretability Analysis for {image_path}...")
    hist_joint, ref_path, curvatures = run_and_record("joint", image_path, model_path)

    steps = np.arange(len(hist_joint["x"]))
    wp_indices = np.array(hist_joint["waypoint_idx"])
    N_curv = len(curvatures)
    matched_k = np.array([abs(curvatures[idx % N_curv]) if N_curv > 0 else 0.0 for idx in wp_indices])

    # Normalize curvature for visual overlay (matching Fig. 8b)
    k_max = np.max(matched_k) if len(matched_k) > 0 and np.max(matched_k) > 0 else 1.0
    k_norm = matched_k / k_max

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Lookahead Ld evolution vs Curvature (Fig. 8b top)
    axes[0].plot(steps, hist_joint["lookahead"], color="#2980b9", linewidth=2.2, label=r"Learned Lookahead $L_d$ (m)")
    axes[0].plot(steps, k_norm * 1.8 + 0.3, color="#e74c3c", linestyle="--", alpha=0.55, label=r"Normalized Path Curvature $|\kappa|$ (dashed)")
    axes[0].set_ylabel(r"Lookahead $L_d$ [m]", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].set_title("Along-Lap Policy Evolution: Dynamic Parameter Scheduling vs Track Curvature (Fig. 8b)", fontsize=12)

    # 2. Steering Gain g evolution (Fig. 8b bottom)
    axes[1].plot(steps, hist_joint["gain"], color="#8e44ad", linewidth=2.2, label=r"Learned Steering Gain $g$")
    axes[1].set_ylabel(r"Steering Gain $g$", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", framealpha=0.9)

    # 3. Speed & Tracking Error
    axes[2].plot(steps, hist_joint["speed"], color="#27ae60", linewidth=2.0, label="Vehicle Speed (m/s)")
    axes[2].plot(steps, hist_joint["tracking_error"], color="#d35400", linewidth=1.8, label="Tracking Error (m)")
    axes[2].set_ylabel("Speed / Error", fontsize=11)
    axes[2].set_xlabel("Control Step Index", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Interpretability Analysis Plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="path1.jpeg")
    parser.add_argument("--model", default="RL_PP_JOINT_MODEL.zip")
    parser.add_argument("--out_trace", default="complete_model_trace.png")
    parser.add_argument("--out_interp", default="rl_pp_interpretability.png")
    args = parser.parse_args()

    plot_trajectory_overlay(args.image, args.model, args.out_trace)
    plot_interpretability(args.image, args.model, args.out_interp)


if __name__ == "__main__":
    main()
