"""
Visualization script to generate high-resolution trajectory overlay plots
for all three racetracks (Hockenheim, Montreal, Yas Marina) using the PPO model.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from stable_baselines3 import PPO, SAC
from bead_gym_env import BeadTraceEnv
from geometry_utils import points_to_polyline_distance


def resolve_path(p: str | Path, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    p_str = str(p)
    for candidate in [
        p_str,
        str(_root_dir / p_str),
        *[str(_root_dir / folder / Path(p_str).name) for folder in subfolders],
        *[f"{folder}/{Path(p_str).name}" for folder in subfolders],
    ]:
        if Path(candidate).exists():
            return str(candidate)
    return p_str


def load_model(model_path: str):
    resolved = resolve_path(model_path)
    try:
        return PPO.load(resolved, device="cpu")
    except Exception:
        return SAC.load(resolved, device="cpu")


def plot_track_trajectory(model, track_img: str, ax, seed: int = 15000):
    img_resolved = resolve_path(track_img)
    env = BeadTraceEnv(image_path=img_resolved, seed=seed)
    obs, _ = env.reset()
    core = env._env
    path = np.asarray(core.path_points, float)
    breaks = set(core.path_breaks)

    traj = [(core.pos_x, core.pos_y)]
    done = trunc = False
    info = {}

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        traj.append((core.pos_x, core.pos_y))

    traj = np.asarray(traj, float)
    d_ref = points_to_polyline_distance(path, traj)
    d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)

    covered = d_ref <= 0.10
    geom10 = float(np.mean(covered)) * 100.0
    geom05 = float(np.mean(d_ref <= 0.05)) * 100.0
    seq = float(info.get("coverage", 0.0)) * 100.0
    rmse = float(np.sqrt(np.mean(d_traj**2)))

    # Plot Reference Contour & Trajectory
    ax.plot(path[:, 0], path[:, 1], color="#7f8c8d", linewidth=3.0, alpha=0.8, label="Reference Track")
    ax.plot(traj[:, 0], traj[:, 1], color="#2980b9", linewidth=2.0, label="PPO Trajectory")
    ax.scatter(path[covered, 0], path[covered, 1], color="#27ae60", s=18, alpha=0.9, label="On-Path (<=0.10m)")
    if np.any(~covered):
        ax.scatter(path[~covered, 0], path[~covered, 1], color="#e74c3c", s=30, marker="x", label="Deviation (>0.10m)")

    ax.scatter([traj[0, 0]], [traj[0, 1]], color="#2ecc71", s=110, marker="o", edgecolors="black", zorder=6, label="Start")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="#e67e22", s=130, marker="X", edgecolors="black", zorder=6, label="Finish")

    track_title = Path(track_img).stem.replace("_track", "").title()
    ax.set_title(f"{track_title} | Precision: {geom10:.1f}% | RMSE: {rmse:.4f}m | Lap: {seq:.1f}%", fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=8)
    env.close()


def main():
    p = argparse.ArgumentParser(description="Plot PPO Trajectories for All Maps")
    p.add_argument("--model", default="results/models/PPO_BTP_MODEL.zip", help="PPO Model checkpoint")
    p.add_argument(
        "--tracks",
        nargs="+",
        default=["hockenheim_track.png", "montreal_track.png", "yasmarina_track.png"],
        help="List of tracks to visualize",
    )
    p.add_argument("--out", default="results/plots/ppo_all_maps_trajectories.png", help="Output PNG path")
    args = p.parse_args()

    model = load_model(args.model)
    fig, axes = plt.subplots(1, len(args.tracks), figsize=(6 * len(args.tracks), 6), squeeze=False)
    axes = axes[0]

    for i, track in enumerate(args.tracks):
        plot_track_trajectory(model, track, axes[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Visualization saved to: {out_path}")


if __name__ == "__main__":
    main()
