import cv2
from matplotlib import pyplot as plt

img = cv2.imread('house.jpg', 0)

img_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
img_canny = cv2.Canny(img, 100, 200)
img_laplacian = cv2.Laplacian(img,cv2.CV_64F)

plt.subplot(231)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(232)
plt.title("Sobel filter")
plt.imshow(img_sobel, cmap='gray')
plt.subplot(233)
plt.title("Canny filter")
plt.imshow(img_canny, cmap='gray')
plt.subplot(234)
plt.title("Laplacian filter")
plt.imshow(img_laplacian, cmap='gray')

plt.tight_layout()
plt.show()
