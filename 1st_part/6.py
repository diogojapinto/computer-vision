import cv2
from matplotlib import pyplot as plt

img = cv2.imread('noisy.jpg')
img_mean = cv2.blur(img,(5,5))
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
img_median = cv2.medianBlur(img, 5)
img_bilateral = cv2.bilateralFilter(img, 9, 75, 75) # preserves edges

plt.subplot(231)
plt.title("Original")
plt.imshow(img)
plt.subplot(232)
plt.title("Mean Filter")
plt.imshow(img_mean)
plt.subplot(233)
plt.title("Gaussian Filter")
plt.imshow(img_gaussian)
plt.subplot(234)
plt.title("Median Filter")
plt.imshow(img_median)
plt.subplot(235)
plt.title("Bilateral Filter")
plt.imshow(img_bilateral)

plt.tight_layout()
plt.show()
