"""
Visualization Suite for Research Paper Tracks:
- Hockenheim (Training layout - Paper Fig. 5a)
- Montreal (Zero-shot test layout - Paper Fig. 5b, Table II)
- Yas Marina (Zero-shot test layout - Paper Fig. 5c, Table III)
"""

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


def resolve_path(p, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p


def run_track(image_path: str, model_path: str = "RL_PP_JOINT_MODEL.zip"):
    image_path = resolve_path(image_path)
    model_path = resolve_path(model_path)
    env = RLPurePursuitEnv(image_path=image_path, mode="joint", seed=42)
    model = PPO.load(model_path, device="cpu")
    obs, info = env.reset()
    done = trunc = False

    history = {
        "x": [env._env.pos_x],
        "y": [env._env.pos_y],
        "speed": [env.speed],
        "lookahead": [env.controller.L_smooth],
        "gain": [env.controller.g_smooth],
        "error": [0.0]
    }

    while not (done or trunc):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(act)
        history["x"].append(env._env.pos_x)
        history["y"].append(env._env.pos_y)
        history["speed"].append(env.speed)
        history["lookahead"].append(info["lookahead_distance"])
        history["gain"].append(info["steering_gain"])
        history["error"].append(info["tracking_error"])

    ref_path = np.asarray(env._env.path_points, dtype=np.float64)
    traj = np.stack([history["x"], history["y"]], axis=1)

    d_ref = points_to_polyline_distance(ref_path, traj)
    geom10 = float(np.mean(d_ref <= 0.10))
    d_traj = points_to_polyline_distance(traj, ref_path)
    rmse = float(np.sqrt(np.mean(d_traj ** 2)))
    seq = float(info["sequential"])

    env.close()
    return ref_path, traj, history, seq, geom10, rmse, len(traj) - 1


def generate_paper_tracks_figure(model_path: str = "RL_PP_JOINT_MODEL.zip", out_path: str = "results/plots/paper_tracks_visualization.png"):
    model_path = resolve_path(model_path)
    tracks = [
        ("Hockenheim", resolve_path("hockenheim_track.png"), "Training Track (Fig. 5a)"),
        ("Montreal", resolve_path("montreal_track.png"), "Zero-Shot Test Track (Fig. 5b / Table II)"),
        ("Yas Marina", resolve_path("yasmarina_track.png"), "Zero-Shot Test Track (Fig. 5c / Table III)")
    ]

    print("=" * 85)
    print("EVALUATING AND VISUALIZING RESEARCH PAPER RACETRACKS (Elgouhary & El-Wakeel 2026)")
    print(f"Model: {model_path}")
    print("=" * 85)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    for idx, (name, img_path, subtitle) in enumerate(tracks):
        print(f"\nProcessing {name} ({img_path})...")
        ref, traj, hist, seq, g10, rmse, steps = run_track(img_path, model_path)

        ax = axes[idx]
        # Reference track
        ax.plot(ref[:, 0], ref[:, 1], color="#7f8c8d", linestyle="--", linewidth=2.5, label="Reference Track", zorder=2)
        # Agent bead trail
        ax.plot(traj[:, 0], traj[:, 1], color="#2980b9", linewidth=2.2, label="RL-PP Bead Trail", zorder=3)
        # Start & Finish markers
        ax.scatter([ref[0, 0]], [ref[0, 1]], color="#27ae60", s=130, marker="o", label="Start Point", zorder=5)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="#c0392b", s=140, marker="X", label="Finish Point", zorder=5)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Arena X (m)", fontsize=10)
        ax.set_ylabel("Arena Y (m)", fontsize=10)
        ax.set_title(
            f"( {chr(97+idx)} ) {name}\n{subtitle}\n"
            f"Sequential: {seq*100:.1f}% | Geom@0.10: {g10*100:.1f}% | RMSE: {rmse:.4f}m",
            fontsize=11, pad=10
        )
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        print(f"  -> {name} Result: Sequential={seq:.2%}, Geom@0.10={g10:.2%}, RMSE={rmse:.4f}, Steps={steps}")

    fig.suptitle(
        "Zero-Shot Tracking Across Research Paper Tracks with RL-Tuned Pure Pursuit (PPO)\n"
        "Joint Lookahead & Steering-Gain Adaptation (Mohamed Elgouhary & Amr S. El-Wakeel, 2026)",
        fontsize=13, y=1.03
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] Multi-Track Comparison Figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RL_PP_JOINT_MODEL.zip")
    parser.add_argument("--out", default="paper_tracks_visualization.png")
    args = parser.parse_args()

    generate_paper_tracks_figure(args.model, args.out)


if __name__ == "__main__":
    main()
