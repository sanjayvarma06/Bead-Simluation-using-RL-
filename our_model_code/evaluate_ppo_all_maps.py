"""
Multi-Map Evaluation Suite for PPO Model across all three racetracks:
- Hockenheim
- Montreal
- Yas Marina
(and path1 / custom tracks)
Verifies precision (Geometric Coverage @ 0.10m) > 95% for every episode.
"""
from __future__ import annotations
import sys
import argparse
import csv
import json
from pathlib import Path
import numpy as np

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


def load_policy(model_path: str):
    resolved = resolve_path(model_path)
    try:
        return PPO.load(resolved, device="cpu")
    except Exception:
        return SAC.load(resolved, device="cpu")


def run_episode(model, image_path: str, seed: int) -> dict:
    resolved_img = resolve_path(image_path)
    env = BeadTraceEnv(image_path=resolved_img, seed=seed)
    obs, _ = env.reset()
    core = env._env
    path = np.asarray(core.path_points, float)
    breaks = set(core.path_breaks)

    traj = [(core.pos_x, core.pos_y)]
    total_reward = 0.0
    done = trunc = False
    info = {}

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        total_reward += float(r)
        traj.append((core.pos_x, core.pos_y))

    traj = np.asarray(traj, float)
    dr = points_to_polyline_distance(path, traj)
    dt = points_to_polyline_distance(traj, path, path_breaks=breaks)

    metrics = {
        "track": Path(image_path).name,
        "seed": seed,
        "reward": float(total_reward),
        "sequential": float(info.get("coverage", 0.0)),
        "geom005": float(np.mean(dr <= 0.05)),
        "geom010": float(np.mean(dr <= 0.10)),
        "geom015": float(np.mean(dr <= 0.15)),
        "rmse": float(np.sqrt(np.mean(dt**2))),
        "mean_error": float(np.mean(dt)),
        "p95_error": float(np.percentile(dt, 95)),
        "max_error": float(np.max(dt)),
        "steps": len(traj) - 1,
    }
    env.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Map PPO Evaluation Suite")
    parser.add_argument("--model", default="results/models/PPO_BTP_MODEL.zip", help="Path to trained PPO model")
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["hockenheim_track.png", "montreal_track.png", "yasmarina_track.png"],
        help="Tracks to evaluate",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per track")
    parser.add_argument("--csv", default="results/metrics_and_logs/ppo_all_maps_evaluation.csv", help="Output CSV path")
    parser.add_argument("--json", default="results/metrics_and_logs/ppo_all_maps_summary.json", help="Output JSON path")
    args = parser.parse_args()

    model = load_policy(args.model)
    print("=" * 95)
    print(f"PPO MULTI-MAP BENCHMARK EVALUATION (Model: {args.model})")
    print(f"Tracks: {args.tracks} | Episodes per track: {args.episodes}")
    print("=" * 95)
    print(f"{'Track Name':<22} | {'Ep':<3} | {'Sequential':<11} | {'Geom@0.10':<11} | {'Geom@0.05':<11} | {'RMSE':<9} | {'P95':<9} | {'Steps'}")
    print("-" * 95)

    all_rows = []
    track_summaries = {}

    for track in args.tracks:
        track_rows = []
        for ep in range(args.episodes):
            seed = 50000 + ep * 137
            row = run_episode(model, track, seed)
            all_rows.append(row)
            track_rows.append(row)
            print(
                f"{row['track']:<22} | {ep+1:02d}  | {row['sequential']:<11.2%} | {row['geom010']:<11.2%} | "
                f"{row['geom005']:<11.2%} | {row['rmse']:<9.4f} | {row['p95_error']:<9.4f} | {row['steps']}"
            )

        # Track stats
        g10 = np.array([r["geom010"] for r in track_rows])
        seq = np.array([r["sequential"] for r in track_rows])
        rmse = np.array([r["rmse"] for r in track_rows])
        p95 = np.array([r["p95_error"] for r in track_rows])

        track_summaries[Path(track).name] = {
            "geom010_mean": float(g10.mean()),
            "geom010_min": float(g10.min()),
            "sequential_mean": float(seq.mean()),
            "rmse_mean": float(rmse.mean()),
            "p95_mean": float(p95.mean()),
            "all_above_95": bool(g10.min() >= 0.95),
        }

    print("=" * 95)
    print("TRACK-BY-TRACK SUMMARY & PRECISION (>95% TARGET VERIFICATION):")
    print("-" * 95)
    for tname, stat in track_summaries.items():
        status = "PASSED (>95%)" if stat["all_above_95"] else f"WARNING (Min: {stat['geom010_min']:.2%})"
        print(
            f"  {tname:<22}: Precision (Geom@0.10) = {stat['geom010_mean']:.2%} (Min: {stat['geom010_min']:.2%}) | "
            f"Seq = {stat['sequential_mean']:.2%} | RMSE = {stat['rmse_mean']:.4f}m -> [{status}]"
        )

    # Save CSV
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Save JSON
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(track_summaries, f, indent=2)

    print(f"\nSaved CSV -> {args.csv}")
    print(f"Saved JSON -> {args.json}")


if __name__ == "__main__":
    main()
