import os
import urllib.request
from pathlib import Path

tracks = {
    "Hockenheim": [
        "Hockenheim_map.png",
        "Hockenheim_centerline.csv",
        "Hockenheim_raceline.csv"
    ],
    "Montreal": [
        "Montreal_map.png",
        "Montreal_centerline.csv"
    ],
    "YasMarina": [
        "YasMarina_map.png",
        "YasMarina_centerline.csv",
        "YasMarina_raceline.csv"
    ]
}

dest_dir = Path("research_paper_tracks")
dest_dir.mkdir(exist_ok=True)

base_url = "https://raw.githubusercontent.com/f1tenth/f1tenth_racetracks/master"

for track, files in tracks.items():
    print(f"\nDownloading {track} files from f1tenth/f1tenth_racetracks...")
    for f in files:
        url = f"{base_url}/{track}/{f}"
        target = dest_dir / f
        try:
            print(f"  -> {f}")
            urllib.request.urlretrieve(url, target)
        except Exception as e:
            print(f"Failed to download {f}: {e}")

print("\nAll research paper tracks downloaded successfully into research_paper_tracks/")
