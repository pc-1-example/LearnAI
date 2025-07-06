import cv2

img = cv2.imread('img1.png')
img = cv2.resize(img, (700, 700)) #Для удобства просмотра

#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(img)
img = cv2.merge([r, g, b])

cv2.imshow('Image', img)
cv2.waitKey(0)