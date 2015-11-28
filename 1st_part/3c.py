import cv2
import sys
import numpy as np

if len(sys.argv) < 3 and not sys.argv[1].endswith('.jpg'):
    print("usage: python 3c.py <jpg-filepath> <channel> <value>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
channel = sys.argv[2]
value = sys.argv[3]

cv2.imshow('1', img[:, :, 0])
cv2.imshow('2', img[:, :, 1])
cv2.imshow('3', img[:, :, 2])

img[:, :, channel] = np.add(img[:, :, channel], np.full(img.shape[:2], value, dtype='uint8'))

cv2.imshow('new', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
