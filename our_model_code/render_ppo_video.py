"""
High-Resolution Video Simulation Renderer for PPO Bead/Track Tracing
Renders MP4 videos with dynamic telemetry dashboard overlay for all racetracks.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import cv2
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


def load_model(model_path: str):
    resolved = resolve_path(model_path)
    try:
        return PPO.load(resolved, device="cpu")
    except Exception:
        return SAC.load(resolved, device="cpu")


def render_video(
    image_path: str = "hockenheim_track.png",
    model_path: str = "results/models/PPO_BTP_MODEL.zip",
    output_video: str = "results/videos/ppo_hockenheim_trace.mp4",
    fps: int = 30,
    frame_skip: int = 4,
    seed: int = 42,
):
    img_resolved = resolve_path(image_path)
    model_resolved = resolve_path(model_path)

    print("=" * 80)
    print(f"RENDERING PPO SIMULATION VIDEO: {Path(image_path).name}")
    print(f"Model       : {model_resolved}")
    print(f"Output Video: {output_video}")
    print("=" * 80)

    model = load_model(model_resolved)
    env = BeadTraceEnv(image_path=img_resolved, seed=seed)
    obs, info = env.reset()
    core = env._env

    path = np.asarray(core.path_points, float)
    breaks = set(core.path_breaks)

    traj = [(core.pos_x, core.pos_y)]
    speeds = [np.hypot(core.vel_x, core.vel_y)]
    errors = [float(info.get("tracking_error", 0.0))]
    coverages = [float(info.get("coverage", 0.0))]

    done = trunc = False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        traj.append((core.pos_x, core.pos_y))
        speeds.append(np.hypot(core.vel_x, core.vel_y))
        errors.append(float(info.get("tracking_error", 0.0)))
        coverages.append(float(info.get("coverage", 0.0)))

    traj = np.asarray(traj, float)
    speeds = np.asarray(speeds, float)
    errors = np.asarray(errors, float)
    coverages = np.asarray(coverages, float)

    d_ref = points_to_polyline_distance(path, traj)
    d_traj = points_to_polyline_distance(traj, path, path_breaks=breaks)
    geom10 = float(np.mean(d_ref <= 0.10)) * 100.0
    rmse = float(np.sqrt(np.mean(d_traj**2)))

    # Setup Canvas Dimensions
    W, H = 1000, 900
    margin_x, margin_y = 70, 80
    top_hud_h = 100

    xmin = min(path[:, 0].min(), traj[:, 0].min()) - 0.6
    xmax = max(path[:, 0].max(), traj[:, 0].max()) + 0.6
    ymin = min(path[:, 1].min(), traj[:, 1].min()) - 0.6
    ymax = max(path[:, 1].max(), traj[:, 1].max()) + 0.6

    scale = min((W - 2 * margin_x) / max(1e-4, (xmax - xmin)), (H - top_hud_h - 2 * margin_y) / max(1e-4, (ymax - ymin)))

    def to_pixel(pt):
        px = int(margin_x + (pt[0] - xmin) * scale)
        py = int(H - margin_y - (pt[1] - ymin) * scale)
        return (px, py)

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    if not writer.isOpened():
        # Fallback to alternative fourcc if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    track_name = Path(image_path).stem.replace("_track", "").upper()

    frame_indices = list(range(0, len(traj), max(1, frame_skip)))
    if frame_indices[-1] != len(traj) - 1:
        frame_indices.append(len(traj) - 1)

    print(f"Generating {len(frame_indices)} frames...")

    for f_idx in frame_indices:
        # Dark Modern Aesthetic Background
        frame = np.full((H, W, 3), (18, 22, 28), dtype=np.uint8)

        # Draw Grid Lines
        for gx in range(0, W, 100):
            cv2.line(frame, (gx, top_hud_h), (gx, H), (25, 32, 40), 1)
        for gy in range(top_hud_h, H, 100):
            cv2.line(frame, (0, gy), (W, gy), (25, 32, 40), 1)

        # Draw Target Contour Reference Path
        path_pixels = [to_pixel(pt) for pt in path]
        for i in range(len(path_pixels) - 1):
            if i not in breaks:
                cv2.line(frame, path_pixels[i], path_pixels[i + 1], (100, 110, 125), 4, cv2.LINE_AA)

        # Draw Start Marker
        cv2.circle(frame, path_pixels[0], 7, (46, 204, 113), -1, cv2.LINE_AA)
        cv2.circle(frame, path_pixels[0], 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Trajectory Trail up to current frame
        sub_traj = traj[: f_idx + 1]
        if len(sub_traj) > 1:
            traj_pixels = [to_pixel(pt) for pt in sub_traj]
            for i in range(len(traj_pixels) - 1):
                err = errors[i]
                # Green when error <= 0.08m, transitioning to Yellow/Red on error
                if err <= 0.06:
                    color = (235, 150, 40)   # Cyan / Blue
                elif err <= 0.10:
                    color = (50, 200, 100)   # Green
                else:
                    color = (60, 60, 240)    # Red deviation
                cv2.line(frame, traj_pixels[i], traj_pixels[i + 1], color, 3, cv2.LINE_AA)

        # Current Vehicle Bead Position
        cur_pt = to_pixel(traj[f_idx])
        # Outer Glow
        cv2.circle(frame, cur_pt, 13, (0, 165, 255), 2, cv2.LINE_AA)
        # Inner Bead
        cv2.circle(frame, cur_pt, 7, (0, 215, 255), -1, cv2.LINE_AA)

        # Top HUD Dashboard Bar
        cv2.rectangle(frame, (0, 0), (W, top_hud_h), (12, 15, 20), -1)
        cv2.line(frame, (0, top_hud_h), (W, top_hud_h), (50, 65, 80), 2)

        # HUD Titles and Values
        cur_spd = speeds[f_idx]
        cur_err = errors[f_idx]
        cur_cov = coverages[f_idx] * 100.0

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Line 1: Track title + Status
        cv2.putText(frame, f"CIRCUIT: {track_name}", (25, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CONTROLLER: PPO REINFORCEMENT LEARNING", (450, 35), font, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

        # Line 2: Telemetry Metrics
        hud_text = (
            f"PROGRESS: {cur_cov:5.1f}%   |   "
            f"PRECISION: {geom10:5.1f}%   |   "
            f"SPEED: {cur_spd:4.2f} m/s   |   "
            f"ERROR: {cur_err:5.3f} m   |   "
            f"STEP: {f_idx:4d}"
        )
        cv2.putText(frame, hud_text, (25, 75), font, 0.58, (180, 210, 230), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"[+] Video successfully saved -> {output_video}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Render PPO Trajectory Simulation Videos")
    parser.add_argument("--image", default="hockenheim_track.png", help="Track image")
    parser.add_argument("--model", default="results/models/PPO_BTP_MODEL.zip", help="PPO Model checkpoint")
    parser.add_argument("--out", default="", help="Output MP4 path")
    parser.add_argument("--all_tracks", action="store_true", help="Render videos for all 3 tracks")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()

    if args.all_tracks:
        tracks = [
            ("hockenheim_track.png", "results/videos/ppo_hockenheim_trace.mp4"),
            ("montreal_track.png", "results/videos/ppo_montreal_trace.mp4"),
            ("yasmarina_track.png", "results/videos/ppo_yasmarina_trace.mp4"),
        ]
        for t_img, out_v in tracks:
            render_video(image_path=t_img, model_path=args.model, output_video=out_v, fps=args.fps)
    else:
        out_v = args.out if args.out else f"results/videos/ppo_{Path(args.image).stem}_trace.mp4"
        render_video(image_path=args.image, model_path=args.model, output_video=out_v, fps=args.fps)


if __name__ == "__main__":
    main()
