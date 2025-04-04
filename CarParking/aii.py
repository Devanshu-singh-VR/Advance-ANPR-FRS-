import cv2

face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

img = cv2.imread('image.jpg')
face = face_cas.detectMultiScale(img, 1.1, 4)

print(face)

x, y, w, h = face[0]

cv2.rectangle(img, (x, y), (x + w, y + h), (43, 255, 25), 2)
cv2.imshow('image', img)

k = cv2.waitKey(0)
