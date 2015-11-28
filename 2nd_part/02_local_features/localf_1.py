import cv2
import numpy as np
from matplotlib import pyplot as plt

img_1 = cv2.imread('feup/feup1.png')
img_2 = cv2.imread('feup/feup2.png')

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# detect the features using Harris corner detection
img_1_gray = np.float32(img_1_gray)
pts_1 = cv2.cornerHarris(img_1_gray, 4, 5, 0.04)
# dilate points only for viewing purposes
pts_1 = cv2.dilate(pts_1, None)

img_2_gray = np.float32(img_2_gray)
pts_2 = cv2.cornerHarris(img_2_gray, 4, 5, 0.04)
pts_2 = cv2.dilate(pts_2, None)

img_1[pts_1  > 0.01 * pts_1.max()] = [0, 0, 255]
img_2[pts_2  > 0.01 * pts_2.max()] = [0, 0, 255]

img_1_show = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
img_2_show = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

plt.subplot(121); plt.imshow(img_1_show)
plt.subplot(122); plt.imshow(img_2_show)
plt.tight_layout(); plt.show()
