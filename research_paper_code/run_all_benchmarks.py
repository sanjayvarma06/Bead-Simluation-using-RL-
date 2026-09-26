import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_root_dir / "our_model_code") not in sys.path:
    sys.path.insert(0, str(_root_dir / "our_model_code"))

from stable_baselines3 import PPO
from rl_pure_pursuit_env import RLPurePursuitEnv
from pure_pursuit import compute_friction_speed_profile
from geometry_utils import points_to_polyline_distance

PAPER_DATA = {
    "RL-PP (joint Ld,g)": {"Mean": 9.46, "Std": 0.23, "Min": 9.09, "Max": 9.82},
    "RL-PP (Ld only)": {"Mean": 9.61, "Std": 0.58, "Min": 8.94, "Max": 10.51},
    "Adaptive PP (linear v->Ld)": {"Mean": 9.72, "Std": 0.27, "Min": 9.34, "Max": 10.40},
    "Fixed PP (Ld fixed)": {"Mean": 9.85, "Std": 0.43, "Min": 9.32, "Max": 10.55},
    "MPC raceline tracker": {"Mean": 15.42, "Std": 0.47, "Min": 14.48, "Max": 16.10},
}

def resolve_track(name):
    candidates = [
        name,
        f"datasets/test_images/{name}",
        str(_root_dir / "datasets" / "test_images" / name),
    ]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return name

def run_all(model_path="results/models/RL_PP_JOINT_MODEL.zip", episodes=5, vmax=6.0, out_plot="results/plots/paper_tracks_visualization.png"):
    tracks = [
        ("Hockenheim", resolve_track("hockenheim_track.png"), "Training Track (Fig. 5a)"),
        ("Montreal", resolve_track("montreal_track.png"), "Zero-Shot Test Track (Fig. 5b / Table II)"),
        ("Yas Marina", resolve_track("yasmarina_track.png"), "Zero-Shot Test Track (Fig. 5c / Table III)")
    ]

    controllers = [
        ("joint", "RL-PP (joint Ld,g)"),
        ("ld_only", "RL-PP (Ld only)"),
        ("adaptive", "Adaptive PP (linear v->Ld)"),
        ("fixed", "Fixed PP (Ld fixed)"),
    ]

    model = None
    if Path(model_path).exists():
        model = PPO.load(model_path, device="cpu")

    print("\n" + "=" * 115)
    print(" UNIFIED BENCHMARK & COMPARISON SUITE (Elgouhary & El-Wakeel 2026)")
    print(f" Evaluated Model: {model_path} | Speed Capped at vmax = {vmax} m/s | Episodes per track: {episodes}")
    print("=" * 115)

    all_track_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    for t_idx, (t_name, t_path, subtitle) in enumerate(tracks):
        print(f"\n" + "#" * 115)
        print(f" TRACK {t_idx+1}/3: {t_name.upper()} ({t_path})")
        print("#" * 115)

        track_ctrl_data = {}

        for mode, label in controllers:
            env = RLPurePursuitEnv(image_path=t_path, mode=mode, seed=14000)
            v_prof = compute_friction_speed_profile(
                env.path_curvatures,
                path_points=env._env.path_points,
                is_closed=env.is_closed,
                v_base=vmax,
                v_min=2.0,
                v_max=vmax,
                alpha=2.0,
                a_brake=4.0,
                a_accel=3.0,
            )
            env.speed_profile = v_prof

            ep_rewards = []
            ep_times = []
            ep_geom10 = []
            ep_seq = []
            ep_rmse = []

            for ep in range(episodes):
                obs, info = env.reset(seed=14000 + ep)
                env.speed_profile = v_prof
                done = trunc = False
                tot_r = 0.0
                steps = 0
                while not (done or trunc):
                    if mode in ("joint", "ld_only") and model is not None:
                        act, _ = model.predict(obs, deterministic=True)
                    else:
                        act = np.zeros(2, dtype=np.float32)
                    obs, r, done, trunc, info = env.step(act)
                    tot_r += float(r)
                    steps += 1

                lap_time = steps * env._env.DT
                traj = np.asarray(env.trajectory, dtype=np.float64)
                path = np.asarray(env._env.path_points, dtype=np.float64)
                breaks = set(env._env.path_breaks)

                d_ref = points_to_polyline_distance(path, traj)
                d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)

                ep_rewards.append(tot_r)
                ep_times.append(lap_time)
                ep_geom10.append(float(np.mean(d_ref <= 0.10)))
                ep_seq.append(float(info.get("sequential", 0.0)))
                ep_rmse.append(float(np.sqrt(np.mean(d_traj ** 2))))

            times_arr = np.array(ep_times)
            rew_arr = np.array(ep_rewards)

            track_ctrl_data[label] = {
                "ep_rewards": [round(float(x), 2) for x in ep_rewards],
                "reward_mean": float(rew_arr.mean()),
                "reward_std": float(rew_arr.std()),
                "time_mean": float(times_arr.mean()),
                "time_std": float(times_arr.std()),
                "time_min": float(times_arr.min()),
                "time_max": float(times_arr.max()),
                "geom10": float(np.mean(ep_geom10)),
                "sequential": float(np.mean(ep_seq)),
                "rmse": float(np.mean(ep_rmse)),
            }

            # Plot visual for RL-PP joint
            if mode == "joint":
                ax = axes[t_idx]
                ax.plot(path[:, 0], path[:, 1], color="#7f8c8d", linestyle="--", linewidth=2.5, label="Reference Track", zorder=2)
                ax.plot(traj[:, 0], traj[:, 1], color="#2980b9", linewidth=2.2, label="RL-PP Bead Trail", zorder=3)
                ax.scatter([path[0, 0]], [path[0, 1]], color="#27ae60", s=130, marker="o", label="Start Point", zorder=5)
                ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="#c0392b", s=140, marker="X", label="Finish Point", zorder=5)

                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.set_xlabel("Arena X (m)", fontsize=10)
                ax.set_ylabel("Arena Y (m)", fontsize=10)
                ax.set_title(
                    f"( {chr(97+t_idx)} ) {t_name}\n{subtitle}\n"
                    f"Sequential: {np.mean(ep_seq)*100:.1f}% | Geom@0.10: {np.mean(ep_geom10)*100:.1f}% | RMSE: {np.mean(ep_rmse):.4f}m",
                    fontsize=11, pad=10
                )
                if t_idx == 0:
                    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

            env.close()

        # Print detailed metrics for this track
        print(f"\n1. REWARD & PRECISION METRICS ON {t_name.upper()}:")
        print(f"{'Controller':<26} | {'5-Ep Rewards':<46} | {'Mean Reward':<16} | {'Geom@0.10 (Precision)':<21} | {'RMSE'}")
        print("-" * 125)
        for ctrl_lbl, d in track_ctrl_data.items():
            rew_str = str(d["ep_rewards"])
            print(f"{ctrl_lbl:<26} | {rew_str:<46} | {d['reward_mean']:>7.2f} +/- {d['reward_std']:<5.2f} | {d['geom10']:>19.2%} | {d['rmse']:.4f}m")

        print(f"\n2. LAP TIME COMPARISON (vmax = {vmax} m/s) ON {t_name.upper()} VS RESEARCH PAPER:")
        print(f"{'Controller':<26} | {'Current Mean':<12} | {'Paper Mean':<11} | {'Current Std':<11} | {'Paper Std':<10} | {'Current Min':<11} | {'Paper Min'}")
        print("-" * 105)
        for ctrl_lbl, pdata in PAPER_DATA.items():
            if ctrl_lbl in track_ctrl_data:
                cdata = track_ctrl_data[ctrl_lbl]
                print(f"{ctrl_lbl:<26} | {cdata['time_mean']:>5.2f}s       | {pdata['Mean']:>5.2f}s      | {cdata['time_std']:>5.2f}s       | {pdata['Std']:>5.2f}s     | {cdata['time_min']:>5.2f}s       | {pdata['Min']:>5.2f}s")
            else:
                print(f"{ctrl_lbl:<26} | {'N/A':>5s}       | {pdata['Mean']:>5.2f}s      | {'N/A':>5s}       | {pdata['Std']:>5.2f}s     | {'N/A':>5s}       | {pdata['Min']:>5.2f}s")

        all_track_results[t_name] = track_ctrl_data

    # Save visual figure
    fig.suptitle(
        "Zero-Shot Tracking Across Research Paper Tracks with RL-Tuned Pure Pursuit (PPO)\n"
        "Joint Lookahead & Steering-Gain Adaptation (Mohamed Elgouhary & Amr S. El-Wakeel, 2026)",
        fontsize=13, y=1.03
    )
    fig.tight_layout()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 115)
    print(f" [SUCCESS] ALL 3 TRACKS EVALUATED & VISUALIZATION FIGURE SAVED TO: {out_plot}")
    print("=" * 115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified benchmark runner for all 3 tracks")
    parser.add_argument("--model", default="results/models/RL_PP_JOINT_MODEL.zip", help="PPO Model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--vmax", type=float, default=6.0, help="Max speed cap (m/s)")
    parser.add_argument("--out", default="results/plots/paper_tracks_visualization.png", help="Output plot path")
    args = parser.parse_args()

    run_all(model_path=args.model, episodes=args.episodes, vmax=args.vmax, out_plot=args.out)
