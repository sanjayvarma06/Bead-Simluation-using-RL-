"""
Generates clean, single-stroke reference images for the research paper racetracks:
- Hockenheim
- Montreal
- Yas Marina
directly from their official F1TENTH / TUM raceline centerlines.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def generate_track_image(csv_path: str, out_img_path: str, img_size: int = 1200, line_thickness: int = 7):
    # Read centerline CSV (first two columns are x_m, y_m)
    df = pd.read_csv(csv_path, comment='#', header=None)
    pts = df.iloc[:, :2].values.astype(np.float64)

    # Normalize into [margin, img_size - margin]
    margin = 80
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()

    scale = (img_size - 2 * margin) / max(max_x - min_x, max_y - min_y)

    px = margin + (pts[:, 0] - min_x) * scale
    # Invert Y for image coordinate system
    py = (img_size - margin) - (pts[:, 1] - min_y) * scale

    pixel_pts = np.stack([px, py], axis=1).astype(np.int32)

    # White background with black track stroke (matching path1.jpeg format)
    canvas = np.full((img_size, img_size), 255, dtype=np.uint8)

    # Draw closed loop polyline
    cv2.polylines(canvas, [pixel_pts], isClosed=True, color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(out_img_path, canvas)
    print(f"Generated clean track image: {out_img_path} ({img_size}x{img_size})")


def main():
    tracks = [
        ("Hockenheim", "research_paper_tracks/Hockenheim_centerline.csv", "hockenheim_track.png"),
        ("Montreal", "research_paper_tracks/Montreal_centerline.csv", "montreal_track.png"),
        ("YasMarina", "research_paper_tracks/YasMarina_centerline.csv", "yasmarina_track.png"),
    ]

    for name, csv_f, out_img in tracks:
        if os.path.exists(csv_f):
            generate_track_image(csv_f, out_img)
        else:
            print(f"Warning: {csv_f} not found!")


if __name__ == "__main__":
    main()
