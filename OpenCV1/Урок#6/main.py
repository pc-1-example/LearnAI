import cv2
import numpy as np

photos = cv2.imread('img1.png')
# photos = cv2.resize(photos, (650, 650))

img = np.zeros(photos.shape[:2], dtype=np.uint8)

circle = cv2.circle(img.copy(), (200, 300), 120, 255, -1)
square = cv2.rectangle(img.copy(), (25, 25), (250, 350), 255, -1)

img = cv2.bitwise_and(photos, photos, mask=circle)

cv2.imshow('Image', img)
cv2.waitKey(0)