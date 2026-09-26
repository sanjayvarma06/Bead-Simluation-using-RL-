"""
Dataset Multi-Shape Tracking Visualizer
Evaluates and visualizes the RL-PP agent across a panel of shapes from the dataset.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from rl_pure_pursuit_env import RLPurePursuitEnv
from geometry_utils import points_to_polyline_distance


def evaluate_shape(image_path: str, model: PPO):
    env = RLPurePursuitEnv(image_path=image_path, mode="joint", seed=12345)
    obs, info = env.reset()
    done = trunc = False

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)

    traj = np.asarray(env.trajectory, dtype=np.float64)
    ref_path = np.asarray(env._env.path_points, dtype=np.float64)
    d_ref = points_to_polyline_distance(ref_path, traj)
    geom10 = float(np.mean(d_ref <= 0.10))
    d_traj = points_to_polyline_distance(traj, ref_path)
    rmse = float(np.sqrt(np.mean(d_traj ** 2)))
    seq_cov = float(info["sequential"])
    steps = int(info["steps"])

    env.close()
    return ref_path, traj, seq_cov, geom10, rmse, steps


def plot_dataset_grid(
    images: list[str],
    model_path: str = "RL_PP_JOINT_MODEL.zip",
    out_file: str = "dataset_tracking_grid.png"
):
    print("=" * 80)
    print("GENERATING MULTI-SHAPE DATASET TRACKING GRID")
    print(f"Model: {model_path}")
    print(f"Output: {out_file}")
    print("=" * 80)

    model = PPO.load(model_path, device="cpu")
    valid_images = [img for img in images if Path(img).exists()]

    if not valid_images:
        print("Error: None of the specified images were found.")
        return

    n = len(valid_images)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.2 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, img_path in enumerate(valid_images):
        ax = axes[idx]
        print(f"Evaluating {img_path}...")
        ref, traj, seq, g10, rmse, steps = evaluate_shape(img_path, model)

        # Plot contour and bead trajectory
        ax.plot(ref[:, 0], ref[:, 1], color="#7f8c8d", linestyle="--", linewidth=2.4, label="Target Contour")
        ax.plot(traj[:, 0], traj[:, 1], color="#2980b9", linewidth=2.0, label="RL-PP Bead Trail")
        ax.scatter([ref[0, 0]], [ref[0, 1]], color="#27ae60", s=70, marker="o", zorder=5, label="Start")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="#c0392b", s=80, marker="X", zorder=5, label="Finish")

        name = Path(img_path).stem
        ax.set_title(
            f"{name}\nSeq: {seq*100:.1f}% | Geom@0.10: {g10*100:.1f}% | RMSE: {rmse:.4f}",
            fontsize=11, pad=8
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Zero-Shot RL-Tuned Pure Pursuit across Research Paper Dataset (Elgouhary & El-Wakeel 2026)",
        fontsize=13, y=0.99
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] Dataset Tracking Grid -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot multi-shape dataset evaluation grid")
    parser.add_argument("--model", default="RL_PP_JOINT_MODEL.zip")
    parser.add_argument("--out", default="dataset_tracking_grid.png")
    args = parser.parse_args()

    default_shapes = [
        "path1.jpeg",
        "circle.jpg",
        "triangle.png",
        "test_shape.jpg",
        "test_shape_hollow_star.jpg",
        "closed_shapes_dataset/test/shape_test_0000.jpg",
    ]

    plot_dataset_grid(default_shapes, model_path=args.model, out_file=args.out)


if __name__ == "__main__":
    main()
