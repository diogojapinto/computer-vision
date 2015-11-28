import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('low-contrast.png')

plt.subplot(321)
plt.title("Original")
plt.imshow(img)
plt.subplot(322)
plt.title("Original Histogram")
plt.hist(img.ravel(), 256, [0, 256])

# improve image

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hist_opt = img_hsv.copy()
img_clahe = img_hsv.copy()

img_hist_opt[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
img_clahe[:, :, 2] = clahe.apply(img_hsv[:, :, 2])

img_hist_opt = cv2.cvtColor(img_hist_opt, cv2.COLOR_HSV2BGR)
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_HSV2BGR)

plt.subplot(323)
plt.title("Histogram Optimization")
plt.imshow(img_hist_opt)
plt.subplot(324)
plt.title("Hist. Opt. Histogram")
plt.hist(img_hist_opt.ravel(), 256, [0, 256])
plt.subplot(325)
plt.title("CLAHE Optimization")
plt.imshow(img_clahe)
plt.subplot(326)
plt.title("CLAHE Histogram")
plt.hist(img_clahe.ravel(), 256, [0, 256])


plt.tight_layout()
plt.show()
