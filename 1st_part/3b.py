import cv2
import sys
import random as rnd

if len(sys.argv) < 2 and not sys.argv[1].endswith('.jpg'):
    print("usage: python 3a.py <jpg-filepath>")

img = cv2.imread(sys.argv[1])

height, width, layers = img.shape

for i in range(height):
    for j in range(width):
        for k in range(layers):
            has_effect = rnd.random()

            if has_effect < rnd.random():
                salt_or_pepper = rnd.choice([0, 255])
                img[i, j, k] = salt_or_pepper

cv2.imshow('s&p', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
