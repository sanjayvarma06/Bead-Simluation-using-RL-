import cv2
import numpy as np

# Create a 600x600 white canvas
img = np.ones((600, 600, 3), dtype=np.uint8) * 255

# Generate points for a 5-pointed star
center = (300, 300)
r_outer, r_inner = 220, 110
pts = []

for i in range(10):
    r = r_outer if i % 2 == 0 else r_inner
    angle = i * np.pi / 5
    x = int(center[0] + r * np.sin(angle))
    y = int(center[1] - r * np.cos(angle))
    pts.append([x, y])

pts = np.array(pts, np.int32)

# Convert to a smooth polyline using a dense interpolation & Gaussian blur
# This rounds off the sharp vertex spikes into soft curves
dense_pts = []
num_segments = 50
for i in range(len(pts)):
    p0 = pts[i]
    p1 = pts[(i + 1) % len(pts)]
    for t in np.linspace(0, 1, num_segments, endpoint=False):
        # Optional: Add slight bezier-like interpolation or just let blur handle it
        pass

# Instead, draw a thick polyline and apply severe blur + threshold to round all corners smoothly
rough_canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255
cv2.polylines(rough_canvas, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 0), thickness=15)

# Smooth the sharp corners using a heavy blur and re-thresholding
gray = cv2.cvtColor(rough_canvas, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (31, 31), 0)
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours of the smoothed shape to get a clean, continuous soft outline
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 0), thickness=3)

cv2.imwrite("test_shape_soft_star.jpg", img)
print("Saved soft-curved star image: test_shape_soft_star.jpg")