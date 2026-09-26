import sys
import json
import argparse
import numpy as np
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

def resolve_path(p, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p

def evaluate(image_path="path1.jpeg", episodes=5, model_path="RL_PP_JOINT_MODEL.zip", vmax=6.0):
    image_path = resolve_path(image_path)
    model_path = resolve_path(model_path)
    print("=" * 105)
    print(f"BENCHMARKING 5 EPISODES ON MAP: {image_path}")
    print(f"Raceline Speed Profile Capped at vmax = {vmax} m/s | Evaluated Model: {model_path}")
    print("=" * 105)

    controllers = [
        ("joint", "RL-PP (joint Ld,g)"),
        ("ld_only", "RL-PP (Ld only)"),
        ("adaptive", "Adaptive PP (linear v->Ld)"),
        ("fixed", "Fixed PP (Ld fixed)"),
    ]

    model = None
    if Path(model_path).exists():
        model = PPO.load(model_path, device="cpu")

    results = {}

    for mode, label in controllers:
        env = RLPurePursuitEnv(image_path=image_path, mode=mode, seed=14000)
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
            ep_rewards.append(tot_r)
            ep_times.append(lap_time)

            traj = np.asarray(env.trajectory, dtype=np.float64)
            path = np.asarray(env._env.path_points, dtype=np.float64)
            breaks = set(env._env.path_breaks)
            d_ref = points_to_polyline_distance(path, traj)
            d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)
            ep_geom10.append(float(np.mean(d_ref <= 0.10)))
            ep_rmse.append(float(np.sqrt(np.mean(d_traj ** 2))))

        env.close()

        times_arr = np.array(ep_times)
        rew_arr = np.array(ep_rewards)

        results[label] = {
            "mode": mode,
            "ep_rewards": [round(float(x), 2) for x in ep_rewards],
            "ep_times": [round(float(x), 3) for x in ep_times],
            "reward_mean": float(rew_arr.mean()),
            "reward_std": float(rew_arr.std()),
            "mean": float(times_arr.mean()),
            "std": float(times_arr.std()),
            "min": float(times_arr.min()),
            "max": float(times_arr.max()),
            "geom10": float(np.mean(ep_geom10)),
            "rmse": float(np.mean(ep_rmse)),
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="path1.jpeg", help="Track map or image")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--vmax", type=float, default=6.0, help="Max speed cap (m/s)")
    parser.add_argument("--model", default="RL_PP_JOINT_MODEL.zip", help="Learned PPO model path")
    args = parser.parse_args()

    current_res = evaluate(image_path=args.image, episodes=args.episodes, model_path=args.model, vmax=args.vmax)

    print("\n" + "=" * 105)
    print("1. EPISODE REWARD RESULTS (5 EPISODES RUN ON THE SAME MAP)")
    print("=" * 105)
    for label, data in current_res.items():
        print(f"\nController: {label}")
        for i, (r, t) in enumerate(zip(data["ep_rewards"], data["ep_times"]), start=1):
            print(f"  Episode {i}: Reward = {r:8.2f} | Lap Time = {t:6.3f}s")
        print(f"  -> Summary: Reward Mean = {data['reward_mean']:.2f} ± {data['reward_std']:.2f} | Geom@0.10 = {data['geom10']:.2%} | RMSE = {data['rmse']:.4f}m")

    print("\n" + "=" * 105)
    print("2. REAL-CAR / SIMULATED LAP TIMES (RACELINE SPEED PROFILE CAPPED AT vmax = 6 m/s)")
    print("   Comparison between Current Model and Research Paper Reported Values")
    print("=" * 105)

    header = ["Controller", "Current Mean", "Paper Mean", "Current Std", "Paper Std", "Current Min", "Paper Min", "Current Max", "Paper Max"]
    rows = []
    for ctrl, pdata in PAPER_DATA.items():
        if ctrl in current_res:
            cdata = current_res[ctrl]
            rows.append([
                ctrl,
                f"{cdata['mean']:.2f}s",
                f"{pdata['Mean']:.2f}s",
                f"{cdata['std']:.2f}s",
                f"{pdata['Std']:.2f}s",
                f"{cdata['min']:.2f}s",
                f"{pdata['Min']:.2f}s",
                f"{cdata['max']:.2f}s",
                f"{pdata['Max']:.2f}s",
            ])
        else:
            # Baseline from paper without local RL execution (e.g. MPC raceline tracker)
            rows.append([
                ctrl,
                "N/A",
                f"{pdata['Mean']:.2f}s",
                "N/A",
                f"{pdata['Std']:.2f}s",
                "N/A",
                f"{pdata['Min']:.2f}s",
                "N/A",
                f"{pdata['Paper Max']:.2f}s" if "Paper Max" in pdata else f"{pdata['Max']:.2f}s",
            ])

    print(f"\n{'Controller':<26} | {'Current Mean':<12} | {'Paper Mean':<11} | {'Current Std':<11} | {'Paper Std':<10} | {'Current Min':<11} | {'Paper Min':<10} | {'Current Max':<11} | {'Paper Max'}")
    print("-" * 125)
    for r in rows:
        print(f"{r[0]:<26} | {r[1]:<12} | {r[2]:<11} | {r[3]:<11} | {r[4]:<10} | {r[5]:<11} | {r[6]:<10} | {r[7]:<11} | {r[8]}")
    print("-" * 125)
