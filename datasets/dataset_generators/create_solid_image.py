import cv2
import numpy as np

# Create a 600x600 white canvas
img = np.ones((600, 600, 3), dtype=np.uint8) * 255

# Generate a large 5-pointed star contour
center = (300, 300)
r_outer, r_inner = 280, 150  # Increased size to trigger >20% area ratio
pts = []

for i in range(10):
    r = r_outer if i % 2 == 0 else r_inner
    angle = i * np.pi / 5
    x = int(center[0] + r * np.sin(angle))
    y = int(center[1] - r * np.cos(angle))
    pts.append([x, y])

pts = np.array(pts, np.int32).reshape((-1, 1, 2))

# Use fillPoly instead of polylines to create a solid mass
cv2.fillPoly(img, [pts], color=(0, 0, 0))

cv2.imwrite("test_shape_solid.jpg", img)
print("Saved new solid test image: test_shape_solid.jpg")