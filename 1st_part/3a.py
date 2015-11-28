import cv2
import sys

if len(sys.argv) < 2 and not sys.argv[1].endswith('.jpg'):
    print("usage: python 3a.py <jpg-filepath>")

img = cv2.imread(sys.argv[1])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('color', img)
cv2.imshow('gray', img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(sys.argv[1].split('.')[0] + '-gray.jpg', img_gray)
