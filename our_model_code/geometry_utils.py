from __future__ import annotations
import numpy as np

def valid_segments(polyline: np.ndarray, path_breaks=None):
    p = np.asarray(polyline, dtype=np.float64)
    if len(p) < 2:
        return np.empty((0,2)), np.empty((0,2))
    starts, ends = p[:-1], p[1:]
    if path_breaks:
        keep = np.ones(len(starts), dtype=bool)
        for k in path_breaks:
            k = int(k)
            if 1 <= k < len(p):
                keep[k-1] = False
        starts, ends = starts[keep], ends[keep]
    return starts, ends

def points_to_polyline_distance(points: np.ndarray, polyline: np.ndarray,
                                path_breaks=None, batch_size: int = 256) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    a, b = valid_segments(polyline, path_breaks)
    if len(a) == 0:
        if len(polyline) == 0:
            return np.full(len(pts), np.inf)
        return np.linalg.norm(pts - np.asarray(polyline, dtype=np.float64)[0], axis=1)

    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    ans = np.empty(len(pts), dtype=np.float64)
    for st in range(0, len(pts), batch_size):
        q = pts[st:st+batch_size]
        ap = q[:, None, :] - a[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((q[:, None, :] - proj) ** 2, axis=2)
        ans[st:st+len(q)] = np.sqrt(np.min(d2, axis=1))
    return ans
