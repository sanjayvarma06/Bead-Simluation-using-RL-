import cv2
import numpy as np

# Create a 600x600 white canvas
img = np.ones((600, 600, 3), dtype=np.uint8) * 255

# Generate a clean hollow star outline
center = (300, 300)
r_outer, r_inner = 220, 110
pts = []

for i in range(10):
    r = r_outer if i % 2 == 0 else r_inner
    angle = i * np.pi / 5
    x = int(center[0] + r * np.sin(angle))
    y = int(center[1] - r * np.cos(angle))
    pts.append([x, y])

pts = np.array(pts, np.int32).reshape((-1, 1, 2))
# Draw a clean hollow line matching your training data style
cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=3)

cv2.imwrite("test_shape_hollow_star.jpg", img)
print("Saved hollow star image: test_shape_hollow_star.jpg")