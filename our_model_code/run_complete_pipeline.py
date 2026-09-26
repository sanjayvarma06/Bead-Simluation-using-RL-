from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent

p = argparse.ArgumentParser(description="Run complete High-Precision PPO Pipeline for All Maps")
p.add_argument(
    "--tracks",
    nargs="+",
    default=["hockenheim_track.png", "montreal_track.png", "yasmarina_track.png"],
    help="Tracks for training & evaluation",
)
p.add_argument("--timesteps", type=int, default=40000, help="Training timesteps")
p.add_argument("--model", default="results/models/PPO_BTP_MODEL.zip", help="Target model path")
p.add_argument("--skip_train", action="store_true", help="Skip training and run evaluation only")
a = p.parse_args()

commands = []
if not a.skip_train:
    commands.append([
        sys.executable,
        str(_this_dir / "train_ppo_v9.py"),
        "--tracks", *a.tracks,
        "--timesteps", str(a.timesteps),
        "--save_path", a.model,
    ])

commands.append([
    sys.executable,
    str(_this_dir / "evaluate_ppo_all_maps.py"),
    "--model", a.model,
    "--tracks", *a.tracks,
    "--episodes", "5",
])

for cmd in commands:
    print("\n" + "=" * 88)
    print("RUNNING:", " ".join(cmd))
    print("=" * 88)
    subprocess.run(cmd, check=True)

print("\n" + "=" * 88)
print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY.")
print(f"PPO Model          : {a.model}")
print("Evaluation CSV     : results/metrics_and_logs/ppo_all_maps_evaluation.csv")
print("Evaluation Summary : results/metrics_and_logs/ppo_all_maps_summary.json")
print("=" * 88)
