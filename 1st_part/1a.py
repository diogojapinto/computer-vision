import cv2
import sys

if len(sys.argv) < 2 and not sys.argv[1].endswith('.jpg'):
    print("usage: python 1.py <jpg-filepath>")

img = cv2.imread(sys.argv[1])
cv2.imshow(sys.argv[1], img)

cv2.waitKey(0)
cv2.destroyAllWindows()
