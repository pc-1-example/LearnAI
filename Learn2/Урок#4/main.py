import cv2


cap = cv2.VideoCapture("videos/main.mp4")

while True:
    success, img = cap.read()
    cv2.imshow('Result', img)

    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.Canny(img, 200, 200)

    kernel = 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break