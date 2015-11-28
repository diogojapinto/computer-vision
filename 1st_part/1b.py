import cv2
import sys

if len(sys.argv) < 2 and not sys.argv[1].endswith('.jpg'):
    print("usage: python 1b.py <jpg-filepath>")

img = cv2.imread(sys.argv[1])

cv2.imwrite(sys.argv[1].split('.')[0] + '.bmp', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
