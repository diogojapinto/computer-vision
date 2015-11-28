import cv2
import sys
import numpy as np

if len(sys.argv) < 3 or not sys.argv[1].endswith('.jpg'):
    print("usage: python 3d.py <jpg-filepath> <value>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

value = int(sys.argv[2])

cv2.imshow('1', img[:, :, 0])
cv2.imshow('2', img[:, :, 1])
cv2.imshow('3', img[:, :, 2])

img[:, :, 1] = np.add(img[:, :, 1], np.full(img.shape[:2], value, dtype='uint8'))

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

cv2.imshow('new', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
