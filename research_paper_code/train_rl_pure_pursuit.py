"""
PPO Training Pipeline for RL-Tuned Pure Pursuit
Implements the exact training recipe from Section III.G of:
'Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control with PPO'
(Elgouhary & El-Wakeel, 2026)
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from rl_pure_pursuit_env import RLPurePursuitEnv
from geometry_utils import points_to_polyline_distance


class LinearLRSchedule:
    """Linear decay learning rate schedule: l(f) = l0 * f (Paper Section III.G)."""
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr

    def __call__(self, progress_remaining: float) -> float:
        return self.initial_lr * progress_remaining


class EvaluationAndCheckpointCallback(BaseCallback):
    """
    Monitors true geometric and sequential coverage, logs PPO diagnostics,
    and checkpoints the best policy snapshot.
    """
    def __init__(
        self,
        eval_env_fn,
        eval_freq: int = 4096,
        save_path: str = "RL_PP_JOINT_MODEL",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_score = -np.inf
        self.best_geom10 = 0.0
        self.best_seq = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            env = self.eval_env_fn()
            res = evaluate_policy(self.model, env, episodes=3)
            env.close()

            sc = 0.70 * res["geom010"] + 0.20 * res["sequential"] - 0.10 * res["rmse"]
            if self.verbose:
                print(
                    f"\n[Eval @ step {self.n_calls:,}] "
                    f"Seq Cov: {res['sequential']:.2%} | "
                    f"Geom@0.10: {res['geom010']:.2%} | "
                    f"Geom@0.05: {res['geom005']:.2%} | "
                    f"RMSE: {res['rmse']:.4f} | "
                    f"Mean Err: {res['mean_error']:.4f} | "
                    f"Reward: {res['reward']:.2f}"
                )

            if sc > self.best_score:
                self.best_score = sc
                self.best_geom10 = res["geom010"]
                self.best_seq = res["sequential"]
                self.model.save(self.save_path)
                # Also mirror as COMPLETE_BTP_MODEL
                self.model.save("COMPLETE_BTP_MODEL")
                if self.verbose:
                    print(f"--> Saved new best checkpoint to {self.save_path}.zip & COMPLETE_BTP_MODEL.zip (Score: {sc:.4f})")

        return True


def evaluate_policy(model, env: RLPurePursuitEnv, episodes: int = 5, seed_base: int = 20000) -> dict:
    """Evaluates the model across episodes and computes geometric and tracking metrics."""
    metrics = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        done = False
        trunc = False
        total_r = 0.0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            total_r += float(r)

        traj = np.asarray(env.trajectory, dtype=np.float64)
        path = np.asarray(env._env.path_points, dtype=np.float64)
        breaks = set(env._env.path_breaks)

        # True geometric coverage: reference point -> polyline
        d_ref = points_to_polyline_distance(path, traj)
        # Tracking error: trajectory sample -> reference polyline
        d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)

        metrics.append({
            "reward": total_r,
            "sequential": float(info["sequential"]),
            "coverage": float(info["coverage"]),
            "geom005": float(np.mean(d_ref <= 0.05)),
            "geom010": float(np.mean(d_ref <= 0.10)),
            "geom015": float(np.mean(d_ref <= 0.15)),
            "rmse": float(np.sqrt(np.mean(d_traj ** 2))),
            "mean_error": float(np.mean(d_traj)),
            "p95_error": float(np.percentile(d_traj, 95)),
            "max_error": float(np.max(d_traj)),
            "steps": int(info["steps"]),
            "lookahead_mean": float(info["lookahead_distance"]),
            "gain_mean": float(info["steering_gain"]),
        })

    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


def make_env_creator(image_path: str, mode: str = "joint", seed: int = 42):
    def _thunk():
        return RLPurePursuitEnv(image_path=image_path, mode=mode, seed=seed)
    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Train PPO RL-PP controller according to paper specs")
    parser.add_argument("--image", default="path1.jpeg", help="Path to input image or image dataset folder")
    parser.add_argument("--timesteps", type=int, default=60000, help="Total environment training steps")
    parser.add_argument("--mode", default="joint", choices=["joint", "ld_only"], help="RL-PP mode")
    parser.add_argument("--out", default="RL_PP_JOINT_MODEL", help="Model checkpoint output name")
    parser.add_argument("--lr", type=float, default=2.4e-4, help="Initial learning rate (paper default 2.4e-4)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print(f"TRAINING RL-TUNED PURE PURSUIT (PPO) - Paper Formulation")
    print(f"Image: {args.image}")
    print(f"Mode: {args.mode}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Base Learning Rate: {args.lr}")
    print("=" * 80)

    # 1. Create vectorized environment
    env_creator = make_env_creator(args.image, mode=args.mode, seed=args.seed)
    vec_env = DummyVecEnv([env_creator])
    # VecNormalize for observations and returns (Paper Section III.G)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=30.0)

    # 2. PPO hyperparameters matching Section III.G of the research paper:
    # n_steps = 2048, batch_size = 256, n_epochs = 5, gamma = 0.99, gae_lambda = 0.98,
    # clip_range = 0.2, target_kl = 0.015, ent_coef = 0.02, vf_coef = 0.6, max_grad_norm = 0.7
    lr_schedule = LinearLRSchedule(args.lr)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        target_kl=0.015,
        ent_coef=0.02,
        vf_coef=0.6,
        max_grad_norm=0.7,
        verbose=1,
        seed=args.seed,
        device="cpu"
    )

    # 3. Setup evaluation callback
    eval_callback = EvaluationAndCheckpointCallback(
        eval_env_fn=env_creator,
        eval_freq=2048,
        save_path=args.out,
        verbose=1
    )

    # 4. Train policy
    print("\nStarting PPO optimization...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    # 5. Final evaluation with best checkpoint
    best_model_path = f"{args.out}.zip"
    if Path(best_model_path).exists():
        print(f"\nLoading best checkpoint: {best_model_path}")
        best_model = PPO.load(best_model_path, device="cpu")
    else:
        best_model = model
        best_model.save(args.out)
        best_model.save("COMPLETE_BTP_MODEL")

    eval_env = env_creator()
    final_results = evaluate_policy(best_model, eval_env, episodes=10)
    eval_env.close()

    # Save summary JSON
    summary_path = "rl_pp_training_summary.json"
    Path(summary_path).write_text(json.dumps(final_results, indent=2))
    # Also update complete_model_summary.json for backward compatibility
    Path("complete_model_summary.json").write_text(json.dumps(final_results, indent=2))

    print("\n" + "=" * 80)
    print("RL-PP FINAL TRAINING SUMMARY (10-Episode Mean)")
    print(f"Sequential Coverage : {final_results['sequential']:.2%}")
    print(f"Geometric @ 0.10    : {final_results['geom010']:.2%}")
    print(f"Geometric @ 0.05    : {final_results['geom005']:.2%}")
    print(f"Geometric @ 0.15    : {final_results['geom015']:.2%}")
    print(f"Tracking RMSE       : {final_results['rmse']:.4f}")
    print(f"Mean Error          : {final_results['mean_error']:.4f}")
    print(f"P95 Error           : {final_results['p95_error']:.4f}")
    print(f"Episode Reward      : {final_results['reward']:.2f}")
    print(f"Lookahead Ld Mean   : {final_results['lookahead_mean']:.3f} m")
    print(f"Steering Gain g Mean: {final_results['gain_mean']:.3f}")
    print(f"Saved Checkpoints   : {args.out}.zip & COMPLETE_BTP_MODEL.zip")
    print(f"Summary JSON        : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
