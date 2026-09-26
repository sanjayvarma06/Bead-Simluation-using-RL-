"""
Interactive Track Viewer
Run this to see real-time bead path tracking on any of the research paper tracks:
- Hockenheim
- Montreal
- Yas Marina
"""

import argparse
from simulate_trajectory import simulate

TRACK_MAP = {
    "hockenheim": "hockenheim_track.png",
    "montreal": "montreal_track.png",
    "yasmarina": "yasmarina_track.png",
    "path1": "path1.jpeg"
}

def main():
    parser = argparse.ArgumentParser(description="View live bead tracking on research paper tracks")
    parser.add_argument(
        "--track",
        choices=["hockenheim", "montreal", "yasmarina", "path1", "all"],
        default="hockenheim",
        help="Which track to visualize (hockenheim, montreal, yasmarina, path1, or all)"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None, help="Episode seed (omit for random episode variation)")
    parser.add_argument("--stochastic", action="store_true", help="Sample from policy distribution (realistic action noise)")
    args = parser.parse_args()

    if args.track == "all":
        for t in ["hockenheim", "montreal", "yasmarina"]:
            print(f"\n>>> Starting visualization for: {t.upper()} (Close window when done to see the next track)")
            simulate(image_path=TRACK_MAP[t], show_window=True, fps=args.fps, seed=args.seed, deterministic=not args.stochastic)
    else:
        print(f"\n>>> Starting visualization for: {args.track.upper()}")
        simulate(image_path=TRACK_MAP[args.track], show_window=True, fps=args.fps, seed=args.seed, deterministic=not args.stochastic)

if __name__ == "__main__":
    main()
