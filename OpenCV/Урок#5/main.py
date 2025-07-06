import cv2

img = cv2.imread('img.png')
img = cv2.resize(img, (700, 700)) #Для удобства просмотра

#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)



cv2.imshow('Image', img)
cv2.waitKey(0)