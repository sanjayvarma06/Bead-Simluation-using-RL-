import sys
import argparse, csv, json
from pathlib import Path
import numpy as np

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from stable_baselines3 import PPO, SAC
from bead_gym_env import BeadTraceEnv
from geometry_utils import points_to_polyline_distance

def resolve_path(p, subfolders=("results/models", "datasets/test_images")):
    if not p:
        return p
    for candidate in [p, str(_root_dir / p), *[str(_root_dir / folder / Path(p).name) for folder in subfolders], *[f"{folder}/{Path(p).name}" for folder in subfolders]]:
        if Path(candidate).exists():
            return str(candidate)
    return p

def load_agent(model_path):
    model_path = resolve_path(model_path)
    try:
        return PPO.load(model_path, device="cpu")
    except Exception:
        return SAC.load(model_path, device="cpu")

def episode(model, image, seed):
    image = resolve_path(image)
    env = BeadTraceEnv(image_path=image, seed=seed)
    obs, _ = env.reset()
    core = env._env
    path = np.asarray(core.path_points, float)
    breaks = set(core.path_breaks)
    traj = [(core.pos_x, core.pos_y)]
    total = 0.0
    done = trunc = False
    info = {}
    while not done and not trunc:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(a)
        total += float(r)
        traj.append((core.pos_x, core.pos_y))
    traj = np.asarray(traj, float)
    dr = points_to_polyline_distance(path, traj)
    dt = points_to_polyline_distance(traj, path, path_breaks=breaks)
    row = dict(
        reward=total, sequential=float(info.get("coverage", 0.0)),
        geom005=float(np.mean(dr <= 0.05)), geom010=float(np.mean(dr <= 0.10)),
        geom015=float(np.mean(dr <= 0.15)), geom018=float(np.mean(dr <= 0.18)),
        rmse=float(np.sqrt(np.mean(dt**2))), mean_error=float(np.mean(dt)),
        p95_error=float(np.percentile(dt, 95)), max_error=float(np.max(dt)),
        reference_p95=float(np.percentile(dr, 95)), reference_max=float(np.max(dr)),
        steps=len(traj) - 1
    )
    env.close()
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="COMPLETE_BTP_MODEL.zip")
    p.add_argument("--image", default="path1.jpeg")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--csv", default="results/metrics_and_logs/complete_model_evaluation.csv")
    a = p.parse_args()
    model = load_agent(a.model)
    rows = []
    for i in range(a.episodes):
        r = episode(model, a.image, 12000 + i)
        rows.append(r)
        print(f"Ep {i+1:02d} | seq={r['sequential']:.2%} | geom@.10={r['geom010']:.2%} | "
              f"reward={r['reward']:.2f} | RMSE={r['rmse']:.4f} | P95={r['p95_error']:.4f} | steps={r['steps']}")
    def ms(k):
        x = np.array([r[k] for r in rows], float)
        return float(x.mean()), float(x.std())
    print("\n=== COMPLETE MODEL SUMMARY ===")
    for label, k, fmt in [
        ("Reward", "reward", ".2f"), ("Sequential coverage", "sequential", ".2%"),
        ("Geometric coverage @0.05", "geom005", ".2%"),
        ("Geometric coverage @0.10", "geom010", ".2%"),
        ("Geometric coverage @0.15", "geom015", ".2%"),
        ("Tracking RMSE", "rmse", ".4f"), ("Mean tracking error", "mean_error", ".4f"),
        ("P95 tracking error", "p95_error", ".4f"), ("Max tracking error", "max_error", ".4f"),
        ("Steps", "steps", ".1f")]:
        m, s = ms(k)
        print(f"{label:28s}: {format(m, fmt)} +/- {format(s, fmt)}")
    print(f"Success >=80% geom@0.10    : {np.mean([r['geom010']>=.80 for r in rows]):.2%}")
    print(f"Success >=95% geom@0.10    : {np.mean([r['geom010']>=.95 for r in rows]):.2%}")
    
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summary_path = Path("results/metrics_and_logs/complete_model_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({k: ms(k)[0] for k in rows[0]}, indent=2))
    print(f"CSV -> {a.csv}")

if __name__=="__main__": main()
