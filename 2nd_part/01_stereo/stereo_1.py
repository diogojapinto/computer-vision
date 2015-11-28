import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load both images
img_left = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# obtain the disparity matrix
stereo = cv2.StereoBM_create(numDisparities=80, blockSize=21)
disparity = stereo.compute(img_left, img_right)
disparity_scaled = cv2.convertScaleAbs(disparity, alpha=(np.iinfo(np.uint8).max/np.iinfo(np.uint16).max))

# plot everything 
plt.subplot(221)
plt.title("Left")
plt.imshow(img_left, cmap='gray')
plt.subplot(222)
plt.title("Right")
plt.imshow(img_right, cmap='gray')
plt.subplot(223)
plt.title("Disparity")
plt.imshow(disparity_scaled, cmap='gray')
plt.tight_layout()
plt.show()
