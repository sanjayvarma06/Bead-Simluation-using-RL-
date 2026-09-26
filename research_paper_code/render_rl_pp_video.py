"""
MP4 Video Renderer for RL-Tuned Pure Pursuit
Visualizes real-time path tracking, agent bead, lookahead point, and telemetry.
"""

from __future__ import annotations
import cv2
import sys
import argparse
from pathlib import Path
import numpy as np

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_root_dir / "our_model_code") not in sys.path:
    sys.path.insert(0, str(_root_dir / "our_model_code"))

from stable_baselines3 import PPO
from rl_pure_pursuit_env import RLPurePursuitEnv
from geometry_utils import points_to_polyline_distance


def render_video(
    image_path: str,
    model_path: str = "RL_PP_JOINT_MODEL.zip",
    out_path: str = "complete_model_trace.mp4",
    fps: int = 30,
    frame_skip: int = 4
):
    print(f"Rendering Video for {image_path} with {model_path} -> {out_path}...")
    env = RLPurePursuitEnv(image_path=image_path, mode="joint", seed=16000)
    model = None
    if Path(model_path).exists():
        model = PPO.load(model_path, device="cpu")

    obs, info = env.reset()
    done = False
    trunc = False

    traj = [(env._env.pos_x, env._env.pos_y)]
    lookaheads = [(env.lookahead_pt[0], env.lookahead_pt[1])]
    telemetry = [info]

    while not (done or trunc):
        if model is not None:
            act, _ = model.predict(obs, deterministic=True)
        else:
            act = np.zeros(2, dtype=np.float32)

        obs, r, done, trunc, info = env.step(act)
        traj.append((env._env.pos_x, env._env.pos_y))
        lookaheads.append((env.lookahead_pt[0], env.lookahead_pt[1]))
        telemetry.append(info)

    traj = np.asarray(traj, dtype=np.float64)
    path = np.asarray(env._env.path_points, dtype=np.float64)
    breaks = set(env._env.path_breaks)

    d_ref = points_to_polyline_distance(path, traj)
    geom10 = float(np.mean(d_ref <= 0.10))
    d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)
    rmse = float(np.sqrt(np.mean(d_traj ** 2)))

    W, H = 960, 840
    margin = 70
    xmin, xmax = min(path[:, 0].min(), traj[:, 0].min()), max(path[:, 0].max(), traj[:, 0].max())
    ymin, ymax = min(path[:, 1].min(), traj[:, 1].min()), max(path[:, 1].max(), traj[:, 1].max())
    pad = 0.8
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    scale = min((W - 2 * margin) / (xmax - xmin), (H - 2 * margin - 70) / (ymax - ymin))

    def pix(pt):
        x = int(margin + (pt[0] - xmin) * scale)
        y = int(H - margin - (pt[1] - ymin) * scale)
        return (x, y)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        print("Warning: mp4v codec failed, skipping video generation.")
        return

    frame_ids = list(range(0, len(traj), max(1, frame_skip)))
    if frame_ids[-1] != len(traj) - 1:
        frame_ids.append(len(traj) - 1)

    for idx in frame_ids:
        frame = np.full((H, W, 3), 250, dtype=np.uint8)

        # Draw reference path
        for j in range(len(path) - 1):
            if (j + 1) in breaks:
                continue
            cv2.line(frame, pix(path[j]), pix(path[j + 1]), (180, 180, 180), 2, cv2.LINE_AA)

        # Draw traversed trail
        if idx > 0:
            pts = np.array([pix(q) for q in traj[:idx + 1]], np.int32)
            cv2.polylines(frame, [pts], False, (215, 95, 30), 3, cv2.LINE_AA)

        # Draw lookahead target point
        la_pt = pix(lookaheads[idx])
        cv2.circle(frame, la_pt, 5, (220, 120, 20), -1, cv2.LINE_AA)
        cv2.line(frame, pix(traj[idx]), la_pt, (200, 160, 80), 1, cv2.LINE_AA)

        # Draw current vehicle/bead
        cv2.circle(frame, pix(traj[idx]), 8, (35, 35, 225), -1, cv2.LINE_AA)
        cv2.circle(frame, pix(traj[0]), 6, (30, 165, 30), -1, cv2.LINE_AA)

        # HUD Telemetry
        tel = telemetry[idx]
        cv2.putText(frame, "RL-Tuned Pure Pursuit (Elgouhary & El-Wakeel 2026)", (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Step: {idx}/{len(traj)-1}   Sequential: {tel['sequential']*100:.1f}%   Speed: {tel['speed']:.2f} m/s", (28, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lookahead Ld: {tel['lookahead_distance']:.2f}m   Gain g: {tel['steering_gain']:.2f}   Tracking Error: {tel['tracking_error']:.4f}m", (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (70, 70, 70), 1, cv2.LINE_AA)

        writer.write(frame)

    # Hold final result for 2 seconds
    for _ in range(2 * fps):
        frame = np.full((H, W, 3), 250, dtype=np.uint8)
        for j in range(len(path) - 1):
            if (j + 1) in breaks:
                continue
            cv2.line(frame, pix(path[j]), pix(path[j + 1]), (180, 180, 180), 2, cv2.LINE_AA)
        pts = np.array([pix(q) for q in traj], np.int32)
        cv2.polylines(frame, [pts], False, (215, 95, 30), 3, cv2.LINE_AA)
        cv2.circle(frame, pix(traj[-1]), 8, (35, 35, 225), -1, cv2.LINE_AA)

        cv2.putText(frame, "TRACKING RESULT", (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sequential Coverage: {telemetry[-1]['sequential']*100:.2f}%   Geometric@0.10: {geom10*100:.2f}%", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Tracking RMSE: {rmse:.4f}m   Total Steps: {len(traj)-1}", (28, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 30, 30), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    env.close()
    print(f"[Saved] MP4 Video -> {out_path} (Sequential: {telemetry[-1]['sequential']:.2%}, Geom@0.10: {geom10:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="path1.jpeg")
    parser.add_argument("--model", default="RL_PP_JOINT_MODEL.zip")
    parser.add_argument("--out", default="complete_model_trace.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    render_video(args.image, args.model, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
