import cv2
import sys
from matplotlib import pyplot as plt

if len(sys.argv) < 2 and not sys.argv[1].endswith('.jpg'):
    print("usage: python 1b.py <jpg-filepath>")

img1 = cv2.imread(sys.argv[1])
print(img1.shape)
img2 = img1
img3 = img1.copy()
img2 = cv2.flip(img3, 1)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

cv2.imshow("odih", img1)
plt.subplot(131)
plt.title('img1')
plt.imshow(img1)
plt.subplot(132)
plt.title('img2')
plt.imshow(img2)
plt.subplot(133)
plt.title('img3')
plt.imshow(img3)
plt.tight_layout()
plt.show()
