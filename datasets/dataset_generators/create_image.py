import cv2
import numpy as np

# Create a 600x600 white canvas
img = np.ones((600, 600, 3), dtype=np.uint8) * 255

# Generate a 5-pointed star contour
center = (300, 300)
r_outer, r_inner = 200, 90
pts = []

for i in range(10):
    r = r_outer if i % 2 == 0 else r_inner
    angle = i * np.pi / 5
    x = int(center[0] + r * np.sin(angle))
    y = int(center[1] - r * np.cos(angle))
    pts.append([x, y])

pts = np.array(pts, np.int32).reshape((-1, 1, 2))
cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=6)

cv2.imwrite("test_shape.jpg", img)
print("Saved new test image: test_shape.jpg")