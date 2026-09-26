"""
V8 final refinement: real SAC checkpoint + DAgger actor repair.

Starts from the already RL-trained V7 SAC model. The SAC actor is then repaired
with Dataset Aggregation (DAgger) so it learns the expert action on states that
the learned policy itself visits. This directly targets compounding path-tracking
error without allowing critic drift to destroy the good policy.

Target:
  sequential >= 99%
  true geometric coverage @0.10 >= 95%
  tracking RMSE <= 0.05
  reward within 100..200
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from bead_gym_env import BeadTraceEnv
from geometry_utils import points_to_polyline_distance

def expert_action(env: BeadTraceEnv) -> np.ndarray:
    core = env._env
    wi = min(core.waypoint_idx, len(core.path_points)-1)
    ti = min(wi + 1, len(core.path_points)-1)
    pos = np.array([core.pos_x, core.pos_y], dtype=np.float32)
    vec = core.path_points[ti].astype(np.float32) - pos
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return np.zeros(2, np.float32)
    return np.clip(vec / n, -1.0, 1.0).astype(np.float32)

def actor_action(model: SAC, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        a = model.actor(x, deterministic=True)[0]
    return a.detach().cpu().numpy().astype(np.float32)

def train_actor(model: SAC, obs: np.ndarray, acts: np.ndarray,
                epochs: int, lr: float, batch_size: int = 512):
    actor = model.actor
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-6)
    x = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    y = torch.as_tensor(acts, dtype=torch.float32, device=model.device)
    n = len(obs)
    last = 0.0
    for ep in range(1, epochs+1):
        perm = torch.randperm(n, device=model.device)
        total = 0.0
        for st in range(0, n, batch_size):
            idx = perm[st:st+batch_size]
            pred = actor(x[idx], deterministic=True)
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
        last = total / n
    actor.eval()
    return last

def collect_expert(image: str, episodes: int, seed: int):
    O, A = [], []
    env = BeadTraceEnv(image_path=image, seed=seed)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        while not done and not trunc:
            a = expert_action(env)
            O.append(obs.copy()); A.append(a.copy())
            obs, _, done, trunc, _ = env.step(a)
    env.close()
    return np.asarray(O, np.float32), np.asarray(A, np.float32)

def collect_dagger(model: SAC, image: str, episodes: int, beta: float, seed: int):
    """Visit states with expert/policy mixture, but label EVERY visited state with expert action."""
    O, A = [], []
    env = BeadTraceEnv(image_path=image, seed=seed)
    rollout_stats = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        total = 0.0
        while not done and not trunc:
            ea = expert_action(env)
            pa = actor_action(model, obs)
            O.append(obs.copy()); A.append(ea.copy())
            action = ea if np.random.random() < beta else pa
            obs, r, done, trunc, info = env.step(action)
            total += float(r)
        rollout_stats.append((float(info["coverage"]), total))
    env.close()
    return np.asarray(O, np.float32), np.asarray(A, np.float32), rollout_stats

def run_episode(model: SAC, image: str, seed: int):
    env = BeadTraceEnv(image_path=image, seed=seed)
    obs, _ = env.reset()
    core = env._env
    path = np.asarray(core.path_points, dtype=np.float64)
    breaks = set(core.path_breaks)
    traj = [(float(core.pos_x), float(core.pos_y))]
    total = 0.0
    done = trunc = False
    info = {}
    while not done and not trunc:
        a = actor_action(model, obs)
        obs, r, done, trunc, info = env.step(a)
        total += float(r)
        traj.append((float(core.pos_x), float(core.pos_y)))
    traj = np.asarray(traj, dtype=np.float64)

    # True geometric coverage: each reference point -> closest point on entire trajectory polyline.
    d_ref = points_to_polyline_distance(path, traj)
    # True tracking error: each trajectory sample -> closest segment of reference path.
    d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)

    result = dict(
        reward=total,
        sequential=float(info["coverage"]),
        geom10=float(np.mean(d_ref <= 0.10)),
        geom05=float(np.mean(d_ref <= 0.05)),
        geom15=float(np.mean(d_ref <= 0.15)),
        rmse=float(np.sqrt(np.mean(d_traj**2))),
        mean_error=float(np.mean(d_traj)),
        p95_error=float(np.percentile(d_traj, 95)),
        max_error=float(np.max(d_traj)),
        steps=len(traj)-1,
    )
    env.close()
    return result

def evaluate(model: SAC, image: str, episodes: int, seed: int):
    vals = [run_episode(model, image, seed+i) for i in range(episodes)]
    keys = vals[0].keys()
    return {k: float(np.mean([v[k] for v in vals])) for k in keys}

def score(st):
    # Accuracy dominates model selection.
    return 0.70*st["geom10"] + 0.20*st["sequential"] - 0.10*st["rmse"]

def target_met(st):
    return (
        st["sequential"] >= 0.99 and
        st["geom10"] >= 0.95 and
        st["rmse"] <= 0.05 and
        100.0 <= st["reward"] <= 200.0
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--base", default="FINAL_BTP_MODEL_95pct.zip")
    p.add_argument("--out", default="COMPLETE_BTP_MODEL")
    p.add_argument("--expert_episodes", type=int, default=8)
    p.add_argument("--dagger_episodes", type=int, default=4)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    if not Path(a.base).exists():
        raise FileNotFoundError(f"Base SAC checkpoint not found: {a.base}")

    np.random.seed(2026)
    torch.manual_seed(2026)

    # Load the real SAC model already trained with 30k RL updates.
    env = BeadTraceEnv(image_path=a.image, seed=42)
    model = SAC.load(a.base, env=env, device=a.device)
    print(f"[V8] Loaded RL-trained SAC base: {a.base}")

    # Full-path expert anchor under the corrected 99% environment.
    anchor_o, anchor_a = collect_expert(a.image, a.expert_episodes, 1000)
    agg_o, agg_a = anchor_o.copy(), anchor_a.copy()
    print(f"[Expert anchor] {len(anchor_o):,} state-action pairs")

    # Gentle anchor adaptation first (important because V7 stopped at 95%).
    mse = train_actor(model, agg_o, agg_a, epochs=8, lr=1e-4)
    st = evaluate(model, a.image, episodes=5, seed=3000)
    best_score = score(st)
    model.save(a.out)
    print(f"[Anchor eval] seq={st['sequential']:.2%} geom@0.10={st['geom10']:.2%} "
          f"reward={st['reward']:.2f} RMSE={st['rmse']:.4f} mse={mse:.6f}")
    print(f"[Saved] {a.out}.zip")

    betas = [0.50, 0.25, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00]
    consecutive_hits = 0

    for rd in range(1, a.rounds+1):
        beta = betas[min(rd-1, len(betas)-1)]
        o, ac, roll = collect_dagger(
            model, a.image, a.dagger_episodes, beta, 4000 + rd*100
        )
        agg_o = np.concatenate([agg_o, o], axis=0)
        agg_a = np.concatenate([agg_a, ac], axis=0)

        # DAgger data grows with the policy's own recovery/failure states.
        # Slightly reduce LR in later rounds.
        lr = 1e-4 if rd <= 3 else 5e-5
        mse = train_actor(model, agg_o, agg_a, epochs=6, lr=lr)
        st = evaluate(model, a.image, episodes=6, seed=5000 + rd*100)
        sc = score(st)

        print(
            f"[DAgger {rd}/{a.rounds}] beta={beta:.2f} data={len(agg_o):,} "
            f"seq={st['sequential']:.2%} geom@0.10={st['geom10']:.2%} "
            f"geom@0.05={st['geom05']:.2%} reward={st['reward']:.2f} "
            f"RMSE={st['rmse']:.4f} P95={st['p95_error']:.4f} "
            f"max={st['max_error']:.4f} mse={mse:.6f}"
        )

        if sc > best_score:
            best_score = sc
            model.save(a.out)
            print(f"[New best] {a.out}.zip")

        if target_met(st):
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        # Require two consecutive successful evaluations to avoid a lucky checkpoint.
        if consecutive_hits >= 2:
            print("[V8] Target exceeded twice consecutively. Stopping.")
            break

    best = SAC.load(f"{a.out}.zip", env=env, device=a.device)
    final = evaluate(best, a.image, episodes=20, seed=9000)
    Path("v8_training_summary.json").write_text(json.dumps(final, indent=2))
    print("\n=== V8 FINAL 20-EPISODE VALIDATION ===")
    print(f"Sequential coverage : {final['sequential']:.2%}")
    print(f"Geometric @ 0.10    : {final['geom10']:.2%}")
    print(f"Geometric @ 0.05    : {final['geom05']:.2%}")
    print(f"Geometric @ 0.15    : {final['geom15']:.2%}")
    print(f"Reward              : {final['reward']:.2f}")
    print(f"Tracking RMSE       : {final['rmse']:.4f}")
    print(f"P95 error           : {final['p95_error']:.4f}")
    print(f"Max error           : {final['max_error']:.4f}")
    print(f"Steps               : {final['steps']:.1f}")
    print(f"Model               : {a.out}.zip")
    env.close()

if __name__ == "__main__":
    main()
