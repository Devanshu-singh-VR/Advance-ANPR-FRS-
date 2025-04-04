import cv2 as cv

img = cv.imread('image2.jpg')

img = cv.resize(img, (690, 620))

cv.imwrite('images2.jpg', img)