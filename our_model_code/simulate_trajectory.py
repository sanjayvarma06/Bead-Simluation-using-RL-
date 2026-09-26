import sys
import argparse
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_root_dir / "research_paper_code") not in sys.path:
    sys.path.insert(0, str(_root_dir / "research_paper_code"))

from stable_baselines3 import PPO
from bead_gym_env import BeadTraceEnv

def _get_rl_pure_pursuit_env():
    try:
        import importlib
        if str(_root_dir / "research_paper_code") not in sys.path:
            sys.path.insert(0, str(_root_dir / "research_paper_code"))
        mod = importlib.import_module("rl_pure_pursuit_env")
        return getattr(mod, "RLPurePursuitEnv")
    except Exception:
        return None

def resolve_path(p, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p

def simulate(
    image_path: str = "path1.jpeg",
    model_path: str = "RL_PP_JOINT_MODEL.zip",
    output_gif: str = "results/videos/simulation_trace.gif",
    show_window: bool = False,
    fps: int = 30,
    frame_interval: int = 4,
    seed: Optional[int] = None,
    deterministic: bool = True
):
    image_path = resolve_path(image_path)
    model_path = resolve_path(model_path)
    if not output_gif.startswith("results") and not Path(output_gif).is_absolute():
        output_gif = f"results/videos/{Path(output_gif).name}"
    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)

    run_seed = seed if seed is not None else int(np.random.randint(1, 1_000_000))
    print("=" * 80)
    print("RL-TUNED PURE PURSUIT BEAD VISUALIZATION (PPO)")
    print(f"Input Image : {image_path}")
    print(f"Model Path  : {model_path}")
    print(f"Episode Seed: {run_seed} ({'Reproducible' if seed is not None else 'Randomized Episode'})")
    print(f"Policy Mode : {'Deterministic (Mean Action)' if deterministic else 'Stochastic (Sampled Action)'}")
    print(f"Output GIF  : {output_gif}")
    print("=" * 80)

    print(f"Loading policy from {model_path}...")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception:
        from stable_baselines3 import SAC
        model = SAC.load(model_path, device="cpu")

    obs_shape = model.observation_space.shape[0] if hasattr(model, "observation_space") and model.observation_space else 16
    print(f"Loading environment with image: {image_path} (Model Obs Dim: {obs_shape})...")

    if obs_shape == 16:
        from bead_gym_env import BeadTraceEnv
        env = BeadTraceEnv(image_path=image_path, seed=run_seed)
        obs, info = env.reset()
        core = env._env
        target_path = np.asarray(core.path_points, float)
        agent_path = [(core.pos_x, core.pos_y)]
        lookahead_pts = [(core.pos_x, core.pos_y)]
        velocities = [(np.hypot(core.vel_x, core.vel_y), 0.0)]
        coverages = [float(info.get("coverage", 0.0))]
        errors = [float(info.get("tracking_error", 0.0))]

        print("Running inference to capture bead trajectory...")
        done = trunc = False
        step_count = 0
        total_reward = 0.0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            if not deterministic:
                action = np.clip(action + np.random.normal(0, 0.03, size=2).astype(np.float32), -1.0, 1.0)
            obs, reward, done, trunc, info = env.step(action)

            agent_path.append((core.pos_x, core.pos_y))
            wi = min(core.waypoint_idx, len(core.path_points) - 1)
            lookahead_pts.append((core.path_points[wi, 0], core.path_points[wi, 1]))
            velocities.append((np.hypot(core.vel_x, core.vel_y), 0.0))
            coverages.append(float(info.get("coverage", 0.0)))
            errors.append(float(info.get("tracking_error", 0.0)))

            total_reward += float(reward)
            step_count += 1
    else:
        RLPurePursuitEnvClass = _get_rl_pure_pursuit_env()
        if RLPurePursuitEnvClass is None:
            raise ImportError("Could not import RLPurePursuitEnv from research_paper_code.")
        env = RLPurePursuitEnvClass(image_path=image_path, mode="joint", seed=run_seed)
        obs, info = env.reset(seed=run_seed)
        core = env._env
        target_path = np.asarray(core.path_points, float)
        agent_path = [(core.pos_x, core.pos_y)]
        lookahead_pts = [(env.lookahead_pt[0], env.lookahead_pt[1])]
        velocities = [(env.speed, env.heading)]
        coverages = [float(info.get("sequential", 0.0))]
        errors = [float(info.get("tracking_error", 0.0))]

        print("Running inference to capture bead trajectory...")
        done = trunc = False
        step_count = 0
        total_reward = 0.0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            if not deterministic:
                action = np.clip(action + np.random.normal(0, 0.03, size=2).astype(np.float32), -1.0, 1.0)
            obs, reward, done, trunc, info = env.step(action)

            agent_path.append((core.pos_x, core.pos_y))
            lookahead_pts.append((env.lookahead_pt[0], env.lookahead_pt[1]))
            velocities.append((env.speed, env.heading))
            coverages.append(float(info.get("sequential", 0.0)))
            errors.append(float(info.get("tracking_error", 0.0)))

            total_reward += float(reward)
            step_count += 1

    agent_path = np.array(agent_path)
    lookahead_pts = np.array(lookahead_pts)
    coverages = np.array(coverages)
    errors = np.array(errors)

    overall_on_path = float(np.mean(errors <= 0.10) * 100.0)
    print(f"Inference complete: {step_count} steps.")
    print(f"  -> Lap Progress        : {coverages[-1]*100:.2f}%")
    print(f"  -> On-Path Precision   : {overall_on_path:.2f}%")
    print(f"  -> Mean Tracking Error : {np.mean(errors):.4f} m")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    if len(target_path) > 0:
        ax.plot(target_path[:, 0], target_path[:, 1], color="#95a5a6", linewidth=4, alpha=0.85, label="Target Contour")
        ax.scatter(target_path[0, 0], target_path[0, 1], color="#27ae60", s=140, zorder=5, label="Start Point", marker="*")

    trail_line, = ax.plot([], [], color="#c0392b", linewidth=2.4, label="Bead Trail")
    bead, = ax.plot([], [], color="#2980b9", marker="o", markersize=9, zorder=6, label="Agent Bead")
    la_marker, = ax.plot([], [], color="#d35400", marker="X", markersize=9, zorder=6, label="Lookahead Point")
    la_line, = ax.plot([], [], color="#e67e22", linestyle="--", linewidth=1.4, zorder=4)

    # Dynamic auto-bounds framing around actual track contour
    if len(target_path) > 0:
        all_x = np.concatenate([target_path[:, 0], agent_path[:, 0]])
        all_y = np.concatenate([target_path[:, 1], agent_path[:, 1]])
        margin = 0.8
        ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    else:
        ax.set_xlim(-core.ARENA, core.ARENA)
        ax.set_ylim(-core.ARENA, core.ARENA)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(
        f"RL-Tuned Pure Pursuit Tracking (PPO)\n"
        f"Shape: {Path(image_path).name} | Precision: {overall_on_path:.1f}% | Progress: {coverages[-1]*100:.1f}%",
        fontsize=12, pad=10
    )
    ax.legend(loc="upper right", framealpha=0.92)

    telemetry_box = ax.text(
        0.03, 0.84, "", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
    )

    frame_indices = list(range(0, len(agent_path), max(1, frame_interval)))
    if frame_indices[-1] != len(agent_path) - 1:
        frame_indices.append(len(agent_path) - 1)

    def init():
        trail_line.set_data([], [])
        bead.set_data([], [])
        la_marker.set_data([], [])
        la_line.set_data([], [])
        telemetry_box.set_text("")
        return trail_line, bead, la_marker, la_line, telemetry_box

    def update(frame_idx):
        i = frame_indices[frame_idx]
        trail_line.set_data(agent_path[:i + 1, 0], agent_path[:i + 1, 1])
        bead.set_data([agent_path[i, 0]], [agent_path[i, 1]])
        la_marker.set_data([lookahead_pts[i, 0]], [lookahead_pts[i, 1]])
        la_line.set_data([agent_path[i, 0], lookahead_pts[i, 0]], [agent_path[i, 1], lookahead_pts[i, 1]])

        spd, hdg = velocities[i]
        cov = float(coverages[i] * 100.0)
        err = float(errors[i])

        on_path_so_far = float(np.mean(errors[:i + 1] <= 0.10) * 100.0)
        is_on_contour = err <= 0.10
        status_tag = "ON LINE" if is_on_contour else "CORRECTING"

        telemetry_box.set_text(
            f"Step: {i}/{step_count}\n"
            f"Speed: {spd:.2f} m/s\n"
            f"Tracking Error: {err:.4f} m\n"
            f"Status: {status_tag}\n"
            f"Path Precision: {on_path_so_far:.1f}%\n"
            f"Lap Progress: {cov:.1f}%"
        )
        return trail_line, bead, la_marker, la_line, telemetry_box

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=True
    )

    if output_gif:
        print(f"Saving animation GIF to {output_gif}...")
        anim.save(output_gif, writer=PillowWriter(fps=fps))
        print(f"Animation saved as {output_gif}!")

    if show_window:
        print("Opening interactive visualization window (close window when done)...")
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize bead path tracking through dataset")
    parser.add_argument("--image", default="path1.jpeg", help="Path to image (e.g. path1.jpeg, circle.jpg, closed_shapes_dataset/test/shape_test_0000.jpg)")
    parser.add_argument("--model", default="RL_PP_JOINT_MODEL.zip", help="Path to trained PPO model")
    parser.add_argument("--out", default="simulation_trace.gif", help="Output GIF filename")
    parser.add_argument("--show", action="store_true", help="Display interactive Matplotlib window")
    parser.add_argument("--seed", type=int, default=None, help="Episode random seed (omit for randomized seed)")
    parser.add_argument("--stochastic", action="store_true", help="Sample from policy distribution instead of deterministic mean")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interval", type=int, default=4, help="Frame subsampling interval")
    args = parser.parse_args()

    simulate(
        image_path=args.image,
        model_path=args.model,
        output_gif=args.out,
        show_window=args.show,
        fps=args.fps,
        frame_interval=args.interval,
        seed=args.seed,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()