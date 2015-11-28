import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.jpg')
img1 = img.copy()
img2 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

#cv2.imshow("doih", edges)
#cv2.waitKey(0)

################################################################################

lines = cv2.HoughLines(edges, 1, np.pi/180, 225)
for rho, theta in lines[:, 0, :]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

################################################################################

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
for x1,y1,x2,y2 in lines[:, 0, :]:
    cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

################################################################################

plt.subplot(131)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.title("Hough Transform")
plt.imshow(img1, cmap='gray')
plt.subplot(133)
plt.title("Probabilistic Hough Transform")
plt.imshow(img2, cmap='gray')

plt.tight_layout()
plt.show()
