"""
Re-runs every controller on the three F1TENTH-derived circuits and stores raw
traces + summary metrics used by the paper figures and tables.

Controllers:
  bead_ppo  : our direct-actuation PPO bead policy (results/models/PPO_BTP_MODEL.zip)
  fixed     : Pure Pursuit, fixed (Ld, g)
  adaptive  : Pure Pursuit, linear v -> Ld schedule
  ld_only   : RL-PP policy, Ld channel only (g held fixed)
  joint     : RL-PP policy, joint (Ld, g)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "our_model_code"))
sys.path.insert(0, str(ROOT / "research_paper_code"))

from stable_baselines3 import PPO
from bead_gym_env import BeadTraceEnv
from rl_pure_pursuit_env import RLPurePursuitEnv
from geometry_utils import points_to_polyline_distance
from train_ppo_v9 import get_actor_action

OUT = ROOT / "paper" / "data"
OUT.mkdir(parents=True, exist_ok=True)
TRACKS = {
    "Hockenheim": str(ROOT / "datasets/test_images/hockenheim_track.png"),
    "Montreal": str(ROOT / "datasets/test_images/montreal_track.png"),
    "Yas Marina": str(ROOT / "datasets/test_images/yasmarina_track.png"),
}
EPISODES = 5
SEEDS = [50000 + 137 * k for k in range(EPISODES)]


def geometry_metrics(path, traj, breaks=()):
    dr = points_to_polyline_distance(path, traj)
    dt = points_to_polyline_distance(traj, path, path_breaks=set(breaks))
    return dr, dt, {
        "geom005": float(np.mean(dr <= 0.05)),
        "geom010": float(np.mean(dr <= 0.10)),
        "geom015": float(np.mean(dr <= 0.15)),
        "rmse": float(np.sqrt(np.mean(dt ** 2))),
        "mean_err": float(np.mean(dt)),
        "p95": float(np.percentile(dt, 95)),
        "max_err": float(np.max(dt)),
    }


def run_bead(model, image, seed):
    env = BeadTraceEnv(image_path=image, seed=seed)
    obs, _ = env.reset()
    core = env._env
    traj, acts = [(core.pos_x, core.pos_y)], []
    done = trunc = False
    info, R = {}, 0.0
    while not (done or trunc):
        a = get_actor_action(model, obs)
        acts.append(np.clip(a, -1, 1))
        obs, r, done, trunc, info = env.step(a)
        R += r
        traj.append((core.pos_x, core.pos_y))
    traj = np.asarray(traj)
    path = np.asarray(core.path_points, float)
    dr, dt, m = geometry_metrics(path, traj, core.path_breaks)
    acts = np.asarray(acts)
    m.update({
        "sequential": float(info.get("coverage", 0.0)),
        "steps": len(traj) - 1,
        "time_s": (len(traj) - 1) * core.DT,
        "reward": float(R),
        "smooth": float(np.mean(np.linalg.norm(np.diff(acts, axis=0), axis=1))),
    })
    return m, {"traj": traj, "path": path, "dt": dt}


def run_pp(mode, model, image, seed):
    env = RLPurePursuitEnv(image_path=image, mode=mode, seed=seed)
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    L, G, K, V, D, E = [], [], [], [], [], []
    R, info = 0.0, {}
    while not (done or trunc):
        a = model.predict(obs, deterministic=True)[0] if model is not None else np.zeros(2, np.float32)
        obs, r, done, trunc, info = env.step(a)
        R += r
        n = len(env.path_curvatures)
        i = env.current_idx
        kmax = max(abs(env.path_curvatures[(i + o) % n]) for o in (0, 5, 12))
        L.append(info["lookahead_distance"]); G.append(info["steering_gain"])
        K.append(kmax); V.append(env.speed); D.append(info["steering_angle"]); E.append(info["tracking_error"])
    traj = np.asarray(env.trajectory)
    path = np.asarray(env._env.path_points, float)
    dr, dt, m = geometry_metrics(path, traj, env._env.path_breaks)
    m.update({
        "sequential": float(info["sequential"]),
        "steps": int(info["steps"]),
        "time_s": int(info["steps"]) * env._env.DT,
        "reward": float(R),
        "smooth": float(np.mean(np.abs(np.diff(D)))),
        "Ld_mean": float(np.mean(L)), "Ld_std": float(np.std(L)),
        "g_mean": float(np.mean(G)), "g_std": float(np.std(G)),
        "completed": bool(info.get("terminated", False)),
    })
    return m, {"traj": traj, "path": path, "dt": dt, "L": np.array(L), "G": np.array(G),
               "K": np.array(K), "V": np.array(V), "D": np.array(D)}


def main():
    bead = PPO.load(str(ROOT / "results/models/PPO_BTP_MODEL.zip"), device="cpu")
    rlpp = PPO.load(str(ROOT / "results/models/RL_PP_JOINT_MODEL.zip"), device="cpu")
    controllers = {
        "bead_ppo": lambda img, s: run_bead(bead, img, s),
        "fixed": lambda img, s: run_pp("fixed", None, img, s),
        "adaptive": lambda img, s: run_pp("adaptive", None, img, s),
        "ld_only": lambda img, s: run_pp("ld_only", rlpp, img, s),
        "joint": lambda img, s: run_pp("joint", rlpp, img, s),
    }
    summary = {}
    for tname, img in TRACKS.items():
        summary[tname] = {}
        for cname, fn in controllers.items():
            rows = []
            for k, s in enumerate(SEEDS):
                m, trace = fn(img, s)
                rows.append(m)
                if k == 0:
                    np.savez_compressed(OUT / f"trace_{tname.replace(' ', '')}_{cname}.npz", **trace)
            agg = {}
            for key in rows[0]:
                vals = np.array([r[key] for r in rows], dtype=float)
                agg[key] = {"mean": float(vals.mean()), "std": float(vals.std()),
                            "min": float(vals.min()), "max": float(vals.max())}
            summary[tname][cname] = agg
            print(f"{tname:11s} {cname:9s} geom@0.10={agg['geom010']['mean']:.4f} "
                  f"rmse={agg['rmse']['mean']:.4f} p95={agg['p95']['mean']:.4f} "
                  f"seq={agg['sequential']['mean']:.4f} t={agg['time_s']['mean']:.2f}s", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("saved", OUT / "summary.json")


if __name__ == "__main__":
    main()
